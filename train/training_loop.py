import functools
import gc
import hashlib
import operator
import os
import sys
import time
from argparse import Namespace
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import blobfile as bf
import numpy as np
import psutil
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Literal

import wandb
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.data.dataset import HumanML3D, Text2MotionDatasetV2
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from diffusion import logger
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import (
    LossAwareSampler,
    UniformSampler,
    create_named_schedule_sampler,
)
from diffusion.respace import SpacedDiffusion
from eval import eval_humanact12_uestc, eval_humanml
from model.mdm import MDM
from sample.types import (
    T_DATA_DICT,
    T_IDX_COLLATED_DIFFUSION_SAMPLE,
    T_IN_MEMORY_TRAJ_SAMPLE,
    T_MODEL_KWARGS,
    T_RAW_DIFFUSION_SAMPLE,
    T_RAW_SAMPLE_TUPLE,
    T_SAMPLE_RETURN_FORMAT,
    T_VISUALIZABLE_TRAJ_SAMPLE,
    DiffusionConditioningDict,
    MotionSequenceSampleDict,
    MotionTextSampleDict,
)
from train.train_platforms import TrainPlatform
from utils import dist_util
from utils.model_util import load_model_wo_clip

from sample.generate import loop_generate
from utils.generate_POC_samples_large_scale_MDM import (
    get_motions_idxs_and_captions_with_keyword, get_prompts_from
)
import random
from utils.generate_POC_samples_large_scale_UHC_batch import uhc_correction_batch


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

import json


def parse_resume_step_from_filename(filename: str) -> int:
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir() -> str:
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint() -> None:
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts: torch.Tensor, losses: Dict) -> None:
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


class TrainLoop:
    def __init__(
        self,
        args: Namespace,
        train_platform: TrainPlatform,
        model: MDM,
        diffusion: SpacedDiffusion,
        data: DataLoader,
        util_mode: bool = False,
    ) -> None:
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.num_epochs = 0
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # * dist.get_world_size()
        self.sync_cuda: bool = torch.cuda.is_available()
        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != "cpu":
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = "uniform"
        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_type, diffusion
        )
        self.eval_wrapper = None
        self.eval_data = None
        self.eval_gt_data = None
        self.use_ddp = False
        self.ddp_model = self.model
        self.use_wandb = args.use_wandb

        # when util mode is on (true), we don't actually train
        if util_mode:
            # skip everything after this
            return

        self.lr = args.lr
        self.log_interval = args.log_interval
        self.log_interval_wandb = args.log_interval_wandb
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            # https://github.com/GuyTevet/motion-diffusion-model/issues/121#issuecomment-1536610455
            try:
                self._load_optimizer_state()
            except ValueError:
                # loaded state dict contains a parameter group that doesn't match the size of optimizer's group
                # can happen when fine-tuning on distributed pre-trained models
                # in which case, just ignore this and leave the optimizer as AdamW with no training history
                pass

        if args.dataset in ["kit", "humanml"] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training

            if args.filter_dataset_by_keyword:
                print("get_dataset_loader for self.eval_gt_data, from test.txt...")
                self.eval_gt_data = get_dataset_loader(
                    name=args.dataset,
                    batch_size=args.eval_batch_size,
                    num_frames=None,
                    split=args.eval_split,
                    hml_mode="gt",
                    subset_by_keyword=args.filter_dataset_by_keyword,
                    mini_dataset_dir=args.mini_dataset_dir
                )
                print("get_dataset_loader for gen_loader, from test.txt...")
                gen_loader = get_dataset_loader(
                    name=args.dataset,
                    batch_size=args.eval_batch_size,
                    num_frames=None,
                    split=args.eval_split,
                    hml_mode="eval",
                    subset_by_keyword=args.filter_dataset_by_keyword,
                    mini_dataset_dir=args.mini_dataset_dir
                )

                # make the number of samples to generate no larger than the evaluation set
                # if we want to filter the ground truth
                num_gt = min(len(self.eval_gt_data.dataset), args.eval_num_samples)
                args.eval_num_samples = num_gt

                # take about ~5% subset to evaluate diversity if looking at a
                # filtered dataset
                self.diversity_times = int(num_gt * 0.20)  # 0.05*4000 = 0.02*1000
            else:
                self.eval_gt_data = get_dataset_loader(
                    name=args.dataset,
                    batch_size=args.eval_batch_size,
                    num_frames=None,
                    split=args.eval_split,
                    hml_mode="gt",
                )
                gen_loader = get_dataset_loader(
                    name=args.dataset,
                    batch_size=args.eval_batch_size,
                    num_frames=None,
                    split=args.eval_split,
                    hml_mode="eval",
                )

            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            self.eval_data = {
                "test": lambda: eval_humanml.get_mdm_loader(
                    model,
                    diffusion,
                    args.eval_batch_size,
                    gen_loader,
                    mm_num_samples,
                    mm_num_repeats,
                    gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples,
                    scale=1.0,
                )
            }
            self.time_to_synthesize_new_data = False

        

    def _load_and_sync_parameters(self) -> None:
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            try:
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                # expects checkpoint to have CLIP properties, e.g.
                # "clip_model.positional_embedding", "clip_model.text_projection", "clip_model.logit_scale"...
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
            except RuntimeError:
                # no CLIP params in the checkpoint
                # see: https://github.com/GuyTevet/motion-diffusion-model/issues/121
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
                logger.log(
                    f"loading model from checkpoint (without CLIP): {resume_checkpoint}..."
                )
                state_dict = torch.load(resume_checkpoint, map_location="cpu")

                load_model_wo_clip(self.model, state_dict)
                self.model.to(dist_util.dev())

    def _load_optimizer_state(self) -> None:
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _should_stop_curr_epoch(self) -> bool:

        stop_for_lr_reasons = not (
            not self.lr_anneal_steps
            or (self.step + self.resume_step < self.lr_anneal_steps)
        )

        return stop_for_lr_reasons
    
    def _should_synthesize_new_data(self) -> bool:

        time_to_synthesize_new_data = False
        if self.args.self_consuming_loop_freq > 0 and self.step > 0:
            time_to_synthesize_new_data =  self.step % self.args.self_consuming_loop_freq == 0

        return time_to_synthesize_new_data

    def _handle_log(self, for_epoch: int) -> None:
        if self.use_wandb and (self.step % self.log_interval_wandb == 0):
            # logging in weights and biases
            dct_logs = dict(logger.get_current().name2val)
            dct_logs["epoch"] = for_epoch
            sys.stdout.flush()
            wandb.log(dct_logs)

        if self.step % self.log_interval == 0:
            for k, v in logger.get_current().name2val.items():
                if k == "loss":
                    print(
                        "step[{}]: loss[{:0.5f}]".format(
                            self.step + self.resume_step, v
                        )
                    )
                if k in ["step", "samples"] or "_q" in k:
                    continue
                else:
                    self.train_platform.report_scalar(
                        name=k, value=v, iteration=self.step, group_name="Loss"
                    )

    def _handle_save(self) -> None:
        if self.step % self.save_interval == 0:
            self.save()
            self.model.eval()
            self.evaluate()
            self.model.train()

            # Run for a finite amount of time in integration tests.
            if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                return
            
    def recompute_training_data(self):

        print("inside recompute_training_data()")

        # STEP 1: get all the prompts that we want to imitate
        keyword = ""  # i.e., filter by trivial keyword, so include everything...
        idx_to_caption = get_motions_idxs_and_captions_with_keyword(keyword)
        prompts = get_prompts_from(idx_to_caption)
        num_to_sample = int(self.args.synthetic_augmentation_percent*len(prompts))
        random.seed(self.step)
        prompts_subset = random.sample(prompts, num_to_sample)
        print("Number of prompts we're sampling: ", len(prompts_subset))

        # STEP 2: save the most recent model
        model_path = self.save()
        synthetic_data_version = model_path.split("/")[-1].split(".")[0] # e.g. 'model000000051'

        # STEP 3a: generate motions using the current model
        synthetic_motions_dir = os.path.join(self.args.synthetic_data_dir_parent, synthetic_data_version)
        print("synthesizing new motions into synthetic_motions_dir:", synthetic_motions_dir)
        # loop_generate(prompts_subset, model_path, synthetic_motions_dir, True)
        loop_generate(prompts_subset, self.model, self.diffusion, self.data, model_path, synthetic_motions_dir, True)
        self.model.train()

        # STEP 3b: do UHC-correction
        if self.args.augmentation_type == "imitation_output":
            uhc_correction_batch(
                synthetic_motions_dir, RECOMPUTE_AND_OVERWRITE=True, VIS=False, 
            )
            print("imitated synthetic motions, inside:", synthetic_motions_dir)

        # STEP 4: alter the dataloader to incorporate those examples
        print("building a new dataloader that incorporates motions from:", synthetic_motions_dir)
        data: DataLoader = get_dataset_loader(
            name=self.args.dataset,
            batch_size=self.args.batch_size,
            # num_frames defaults to 60 and is the maximum number of frames to use in
            # training. If training with HumanML3D, this field is ignored
            num_frames=self.args.num_frames,
            subset_by_keyword=self.args.filter_dataset_by_keyword,
            synthetic_data_dir=synthetic_motions_dir,
            synthetic_augmentation_percent=self.args.synthetic_augmentation_percent,
            augmentation_type=self.args.augmentation_type,
        )

        return data

    def run_loop(self) -> None:
        for epoch in range(self.num_epochs):
            print(f"Starting epoch {epoch}")

            if self.time_to_synthesize_new_data: # see logic inside self._should_stop_curr_epoch()
                self.time_to_synthesize_new_data = False
                self.data = self.recompute_training_data()


            for motion, cond, _ in tqdm(self.data):

                if self._should_stop_curr_epoch():
                    break
                if self._should_synthesize_new_data():
                    self.time_to_synthesize_new_data = True

                motion = motion.to(self.device)
                cond["y"] = {
                    key: val.to(self.device) if torch.is_tensor(val) else val
                    for key, val in cond["y"].items()
                }

                self.run_step(motion, cond)

                if self.use_wandb and (self.step % self.log_interval_wandb == 0):
                    # logging in weights and biases
                    dct_logs = dict(logger.get_current().name2val)
                    dct_logs["epoch"] = epoch
                    # print(dct_logs)
                    # sys.stdout.flush()
                    wandb.log(dct_logs)

                if self.step % self.log_interval == 0:
                    for k, v in logger.get_current().name2val.items():
                        if k == "loss":
                            print(
                                "step[{}]: loss[{:0.5f}]".format(
                                    self.step + self.resume_step, v
                                )
                            )

                        if k in ["step", "samples"] or "_q" in k:
                            continue
                        else:
                            self.train_platform.report_scalar(
                                name=k, value=v, iteration=self.step, group_name="Loss"
                            )

                if self.step > 0 and self.step % self.save_interval == 0:

                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1

                if self.time_to_synthesize_new_data:
                    break

            if self._should_stop_curr_epoch():
                break

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self) -> None:
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            print("Running evaluation loop: [Should take about 8.4 min]")
            log_file = os.path.join(
                self.save_dir, f"eval_humanml_{(self.step + self.resume_step):09d}.log"
            )

            if self.args.filter_dataset_by_keyword:
                diversity_times = self.diversity_times  # 14
            else:
                diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training

            eval_dict = eval_humanml.evaluation(
                self.eval_wrapper,
                self.eval_gt_data,
                self.eval_data,
                log_file,
                replication_times=self.args.eval_rep_times,
                diversity_times=diversity_times,
                mm_num_times=mm_num_times,
                run_mm=False,
            )

            # dump the metrics into the eval_dict json
            eval_dict_path = os.path.join(self.args.save_dir, "eval_dict.json")
            historical_eval_metrics = {}
            try:
                with open(eval_dict_path, "r") as f:
                    historical_eval_metrics = json.load(f)
            except IOError:
                print(f"initializing {eval_dict_path}")
            historical_eval_metrics[self.ckpt_file_name()] = {k: v.tolist() for k, v in eval_dict.items()}
            with open(eval_dict_path, "w") as fp:
                json.dump(historical_eval_metrics, fp, indent=4)
            
            if self.use_wandb:
                print(
                    eval_dict
                )  # keys = ['Matching Score_ground truth', 'Matching Score_test', 'R_precision_ground truth', 'R_precision_test', 'FID_ground truth', 'FID_test', 'Diversity_ground truth', 'Diversity_test']

                eval_dict_wandb = {}
                eval_dict_wandb["Matching Score_gt"] = eval_dict[
                    "Matching Score_ground truth"
                ]
                eval_dict_wandb["Matching Score_test"] = eval_dict[
                    "Matching Score_test"
                ]

                eval_dict_wandb["R_precision_gt_0"] = eval_dict[
                    "R_precision_ground truth"
                ][0]
                eval_dict_wandb["R_precision_gt_1"] = eval_dict[
                    "R_precision_ground truth"
                ][1]
                eval_dict_wandb["R_precision_gt_2"] = eval_dict[
                    "R_precision_ground truth"
                ][2]

                eval_dict_wandb["R_precision_test_0"] = eval_dict["R_precision_test"][0]
                eval_dict_wandb["R_precision_test_1"] = eval_dict["R_precision_test"][1]
                eval_dict_wandb["R_precision_test_2"] = eval_dict["R_precision_test"][2]
                eval_dict_wandb["FID_gt"] = eval_dict["FID_ground truth"]
                eval_dict_wandb["FID_test"] = eval_dict["FID_test"]
                eval_dict_wandb["Diversity_gt"] = eval_dict["Diversity_ground truth"]
                eval_dict_wandb["Diversity_test"] = eval_dict["Diversity_test"]

                print(eval_dict_wandb)
                wandb.log(eval_dict_wandb)

            for k, v in eval_dict.items():
                if k.startswith("R_precision"):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(
                            name=f"top{i + 1}_" + k,
                            value=v[i],
                            iteration=self.step + self.resume_step,
                            group_name="Eval",
                        )
                else:
                    self.train_platform.report_scalar(
                        name=k,
                        value=v,
                        iteration=self.step + self.resume_step,
                        group_name="Eval",
                    )

        elif self.dataset in ["humanact12", "uestc"]:
            eval_args = SimpleNamespace(
                num_seeds=self.args.eval_rep_times,
                num_samples=self.args.eval_num_samples,
                batch_size=self.args.eval_batch_size,
                device=self.device,
                guidance_param=1,
                dataset=self.dataset,
                unconstrained=self.args.unconstrained,
                model_path=os.path.join(self.save_dir, self.ckpt_file_name()),
            )
            eval_dict = eval_humanact12_uestc.evaluate(
                eval_args,
                model=self.model,
                diffusion=self.diffusion,
                data=self.data.dataset,
            )
            print(
                f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}'
            )
            for k, v in eval_dict["feats"].items():
                if "unconstrained" not in k:
                    self.train_platform.report_scalar(
                        name=k,
                        value=np.array(v).astype(float).mean(),
                        iteration=self.step,
                        group_name="Eval",
                    )
                else:
                    self.train_platform.report_scalar(
                        name=k,
                        value=np.array(v).astype(float).mean(),
                        iteration=self.step,
                        group_name="Eval Unconstrained",
                    )

        end_eval = time.time()
        print(f"Evaluation time: {round(end_eval-start_eval)/60}min")

    def run_step(
        self, batch: torch.Tensor, cond: Dict[str, DiffusionConditioningDict]
    ) -> None:
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(
        self, batch: torch.Tensor, cond: Dict[str, DiffusionConditioningDict]
    ) -> None:
        self.mp_trainer.zero_grad()

        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self) -> None:
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self) -> None:
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self) -> str:
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self) -> None:
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith("clip_model.")]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()

            model_save_path = bf.join(self.save_dir, filename)

            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

            return model_save_path

        model_save_path = save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)
        
        return model_save_path


class SelfConsumingTrainLoop(TrainLoop):
    # from parser_util. The maximum duration of a sequence that HumanML3D can
    # generate via our diffusion model is 9.8 seconds. The framerate is 20 fps
    # so this is the maximum motion sequence length for a sample.
    HML_MAX_FRAMES: int = int(9.8 * 20)  # = 196

    def __init__(
        self,
        args: Namespace,
        train_platform: TrainPlatform,
        model: MDM,
        diffusion: SpacedDiffusion,
        data: DataLoader,
        util_mode: bool = False,
    ) -> None:
        """
        Find argument details in: utils/parser_util.py
        """
        super().__init__(args, train_platform, model, diffusion, data, util_mode)

        # behavior is different for a2m vs. t2m
        self.is_a2m = args.dataset == "humanact12"

        # --- sampling / generating related args (see generate.py)
        self.uncontrained = False
        self.sampler_batch_size = (
            # default is 64
            args.batch_size
        )

        # configure the dataset for which we will replace samples
        self.dataset_ptr: HumanML3D = self.data.dataset

        self.max_frames: int = 196 if args.dataset in ["kit", "humanml"] else 60
        self.fps: float = 12.5 if args.dataset == "kit" else 20

        # see parser_util.py, motion length default is 6.0
        motion_length = getattr(args, "motion_length", 6.0)
        self.n_frames = min(self.max_frames, int(motion_length * self.fps))

        self.guidance_param: float = args.guidance_param

        if util_mode:
            # skip everything after this
            return

        # get the md5 hash of all the args together, this will be the key for
        # where we save the samples that we generate
        self.args_json_path = os.path.join(args.save_dir, "args.json")
        with open(self.args_json_path, "r") as f:
            args_json = "\n".join(f.readlines())
        self.args_json_md5: str = hashlib.md5(args_json.encode()).hexdigest()

        self.use_wandb = args.use_wandb

        # no conditioning text information
        if getattr(args, "unconstrained", False):
            raise ValueError(
                "Unconstrained training is supported only for HumanAct12. SelfConsuming Loop "
                "supports only HumanML3D."
            )

        # unsupported mode
        if not (
            isinstance(self.data.dataset, HumanML3D)
            and self.data.dataset.mode == "train"
        ):
            raise ValueError(
                "SelfConsuming Loop Expects HumanML3D as its dataset in train mode"
            )

        self.t2m_dataset_ptr: Text2MotionDatasetV2 = self.dataset_ptr.t2m_dataset
        self.t2m_dataset_ptr.set_save_generated_path(args.save_generated_path)

        self.percent_samples_to_replace_per_epoch: float = (
            args.percent_samples_to_replace_per_epoch
        )
        if self.percent_samples_to_replace_per_epoch > 1:
            raise ValueError("Maximum percent replacement is 1!")

        total_samples_in_dataset = len(self.t2m_dataset_ptr)
        self.samples_to_generate_per_epoch = int(
            total_samples_in_dataset * self.percent_samples_to_replace_per_epoch
        )

    def _get_sample_rep(self, model_kwargs: T_MODEL_KWARGS) -> T_RAW_DIFFUSION_SAMPLE:
        """
        A 'rep' is a single run of generation on conditioning information. In the original
        generate script, we have the option to generate n reps for a single batch of conditioning
        information. For example, if we supply 1 text caption and run for 1 rep with batch size 1,
        we will get 1 sample returned.

        If we supply 1 text caption and run for 2 reps with batch size 1, we will get 2 samples
        that are produced given the same conditioning caption.
        """
        # add classifier free guidance scale to batch if it is
        # given to this class
        if self.guidance_param != 1:
            model_kwargs["y"]["scale"] = (
                torch.ones(self.sampler_batch_size, device=dist_util.dev())
                * self.guidance_param
            )

        # configure sampler function of diffusion
        # TODO: how exactly to tweak temperature?
        # self.diffusion is an instance of SpacedDiffusion
        sample_fn = self.diffusion.p_sample_loop

        # sample is a non-differentiable batch of $batch_size number of samples
        # returns:
        #     (batch_size, pose_dim, 1, seq_length)
        #     (batch_size, [263|251], 1, t <= 196),
        # ex: (64, 263, 1, 196)
        return sample_fn(
            # self.model is an instance of MDM
            self.model,
            (
                self.sampler_batch_size,
                self.model.njoints,
                self.model.nfeats,
                self.HML_MAX_FRAMES,
            ),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

    def _process_conditioning_vector(
        self, sample: torch.Tensor, model_kwargs: T_MODEL_KWARGS
    ) -> T_IN_MEMORY_TRAJ_SAMPLE:
        # extract batched conditioning information that was used in generation
        condition: DiffusionConditioningDict = model_kwargs["y"]

        # ('text'|'action_text')
        # HumanML3D (text-to-motion) uses 'text' key
        # HumanAct12 (action-to-motion) uses 'action' key
        if self.args.unconstrained:
            caption = ["unconstrained"] * self.args.num_samples
        else:
            caption: str = (
                condition["text"] if "text" in condition else condition["action_text"]
            )

        # inverse of "_".join(token) for a batch
        # when text-only dataset loaded / generate only, tokens do not exist in conditioning
        # object
        tokens: List[str] = list(
            map(
                lambda x: x.split("_") if x is not None else x,
                condition.get("tokens", []),
            )
        )

        # (batch_size, num_joints, 3, seq_length)
        motions: np.ndarray = sample.cpu().numpy()
        lengths: np.ndarray = condition["lengths"].cpu().numpy()
        sample_obj: T_IN_MEMORY_TRAJ_SAMPLE = (caption, tokens, motions, lengths)
        return sample_obj

    def process_raw_sample_rep(
        self,
        sample: T_RAW_DIFFUSION_SAMPLE,
        model_kwargs: T_MODEL_KWARGS,
        return_format: T_SAMPLE_RETURN_FORMAT,
    ) -> Union[T_IN_MEMORY_TRAJ_SAMPLE, T_VISUALIZABLE_TRAJ_SAMPLE]:
        print(f"data rep: {self.model.data_rep}.....")

        if return_format == "in_memory":
            # * transpose it
            # (batch_size, pose_dim, 1, seq_length) ->
            # (batch_size, 1, seq_length, pose_dim)
            # ex: (64, 1, 196, 263)
            sample = self.data.dataset.t2m_dataset.inv_transform(
                sample.cpu().permute(0, 2, 3, 1)
            ).float()

            # * remove singular dim
            # (batch_size, 1, seq_length, pose_dim) ->
            # (batch_size, seq_length, pose_dim)
            sample = sample.squeeze(1)

            return self._process_conditioning_vector(sample, model_kwargs)

        elif return_format == "save_raw_motion":
            # * transpose it
            # (batch_size, pose_dim, 1, seq_length) ->
            # (batch_size, 1, seq_length, pose_dim)
            # ex: (64, 1, 196, 263)
            sample = self.data.dataset.t2m_dataset.inv_transform(
                sample.cpu().permute(0, 2, 3, 1)
            ).float()

            # * remove singular dim
            # (batch_size, 1, seq_length, pose_dim) ->
            # (batch_size, seq_length, pose_dim)
            sample = sample.squeeze(1)

            np.save(".tmp_generate/hml_motion.npy", sample)

            return self._process_conditioning_vector(sample, model_kwargs)

        elif return_format == "to_visualize":
            # Recover XYZ *positions* from HumanML3D vector representation
            if self.model.data_rep == "hml_vec":
                print("hml vec rep")
                # here, n_joints = 22 if text-to-motion (HumanML3D), 21 if human-act-12 (KIT)
                n_joints = 22 if sample.shape[1] == 263 else 21

                sample = self.data.dataset.t2m_dataset.inv_transform(
                    sample.cpu().permute(0, 2, 3, 1)
                ).float()

                # here, sample.shape = torch.Size([1, 1, 196, 22, 3])
                sample = recover_from_ric(sample, n_joints)

                # here, sample.shape = torch.Size([1, 22, 3, 196])
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            rot2xyz_pose_rep = (
                "xyz"
                if self.model.data_rep in ["xyz", "hml_vec"]
                else self.model.data_rep
            )

            rot2xyz_mask = (
                None
                if rot2xyz_pose_rep in ("xyz", "rot6d")
                else model_kwargs["y"]["mask"]
                .reshape(self.batch_size, self.n_frames)
                .bool()
            )

            sample = self.model.rot2xyz(
                x=sample,
                mask=rot2xyz_mask,
                pose_rep=rot2xyz_pose_rep,
                glob=True,
                translation=True,
                jointstype="smpl",
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False,
            )

            return self._process_conditioning_vector(sample, model_kwargs)

    def process_raw_samples(
        self,
        raw_sample_tuple: T_RAW_SAMPLE_TUPLE,
        return_format: Optional[T_SAMPLE_RETURN_FORMAT] = "in_memory",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_sample, idxs, num_reps, model_kwargs = raw_sample_tuple
        num_samples = num_reps * self.sampler_batch_size

        all_motions: List[np.ndarray] = []
        all_lengths: List[np.ndarray] = []
        all_p_token: List[List[str]] = []
        all_caption: List[str] = []

        for rep_i, sample_rep in enumerate(raw_sample):
            print(f"\tProcessing Sample [repetitions: {rep_i}]")
            sample_obj: T_IN_MEMORY_TRAJ_SAMPLE = self.process_raw_sample_rep(
                sample_rep,
                model_kwargs,
                return_format=return_format,
            )
            caption, tokens, motions, lengths = sample_obj
            all_caption.append(caption)
            all_p_token.append(tokens)
            all_motions.append(motions)
            all_lengths.append(lengths)

        # consolidate all the samples, cut them to be length of total samples
        # all have axis/dimension 0 as the batch dimension
        all_motions = np.concatenate(all_motions, axis=0)[:num_samples]
        all_lengths = np.concatenate(all_lengths, axis=0)[:num_samples]
        all_caption = np.concatenate(all_caption, axis=0)[:num_samples]
        all_p_token = np.concatenate(all_p_token, axis=0)[:num_samples]

        return all_motions, all_lengths, all_caption, all_p_token

    def generate_raw_samples(
        self, model_kwargs: T_MODEL_KWARGS, idxs: torch.Tensor, num_reps: int
    ) -> T_RAW_SAMPLE_TUPLE:
        # the number of samples generated here is equal to:
        #   n = (num_reps * self.sampler_batch_size)
        print(f"Sampler batch size: {self.sampler_batch_size}")
        raw_samples: List[T_RAW_DIFFUSION_SAMPLE] = []

        for rep_i in range(num_reps):
            print(f"\tSampling [repetitions: {rep_i}]")
            sample_rep: T_RAW_DIFFUSION_SAMPLE = self._get_sample_rep(model_kwargs)
            raw_samples.append(sample_rep)

        return raw_samples, idxs, num_reps, model_kwargs

    def format_raw_samples_as_data_dict(
        self, raw_sample_tuple: T_RAW_SAMPLE_TUPLE
    ) -> T_DATA_DICT:
        _, idxs, _, _ = raw_sample_tuple
        all_motions, all_lengths, all_caption, all_p_token = self.process_raw_samples(
            raw_sample_tuple,
            return_format="in_memory",
        )

        # transform this into T_DATA_DICT type so we can replace it in HumanML3D.t2m_dataset.data_dict
        data_dict: T_DATA_DICT = {}

        for motion, caption, tokens, length, idx in zip(
            all_motions, all_caption, all_p_token, all_lengths, idxs
        ):
            datapoint_name = self.t2m_dataset_ptr.get_name_from_item(int(idx))
            text: MotionTextSampleDict = {"caption": caption, "tokens": tokens.tolist()}

            # motion should be: shape: (seq_length, [263|251])
            datapoint: MotionSequenceSampleDict = {
                "motion": motion,
                # length is not necessarily the length of the motion. It is a 'mask' that defines
                # a subsequence of the motion from which we sample. This is how the 'mask' property
                # is generated in the t2m_collate function when __get_item__ is called by dataloader.
                "length": length,
                "text": [text],
            }
            data_dict[datapoint_name] = datapoint

        return data_dict

    def run_step(self, batch: torch.Tensor, cond: T_MODEL_KWARGS) -> None:
        """
        Training of the model occurs here. This is called once per batch in
        an epoch.
        """
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def _unpack_dataset_sample(
        self, sample: T_IDX_COLLATED_DIFFUSION_SAMPLE
    ) -> T_IDX_COLLATED_DIFFUSION_SAMPLE:
        # convert tensors to current device
        motion, cond, idxs = sample
        motion = motion.to(self.device)
        cond["y"]: DiffusionConditioningDict = {
            key: val.to(self.device) if torch.is_tensor(val) else val
            for key, val in cond["y"].items()
        }
        return motion, cond, idxs

    def run_loop(self) -> None:
        """
        Run the training of our model.
        """
        for epoch in range(self.num_epochs):
            print(f"\nStarting epoch {epoch}")

            # from: https://stackoverflow.com/a/21632554
            # print how much memory the program uses
            print(
                "ps memory:"
                + str(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
                + "mb"
            )

            # loads existing generated samples into memory
            # during training of epoch n, load samples generated at the
            # end of epoch n - 1, hence the epoch - 1
            if self.t2m_dataset_ptr.is_generated_samples_already_on_disk(
                epoch - 1, self.args_json_md5
            ):
                raw_sample = self.t2m_dataset_ptr.load_generated_samples_from_disk(
                    epoch - 1, self.args_json_md5
                )["raw_samples"]
                to_replace_in_memory = self.format_raw_samples_as_data_dict(raw_sample)

                self.t2m_dataset_ptr.replace_samples(to_replace_in_memory)

                del raw_sample
                del to_replace_in_memory

            # see: data_loaders/tensors.py
            # collate function to understand how dataset iterator works
            for samp in tqdm(self.data):
                # tqdm shows total 350 if bs = 64, 20,000 / 64 = ~350
                if self._should_stop_curr_epoch():
                    break

                motion, cond, _ = self._unpack_dataset_sample(samp)

                # actually train the model
                self.run_step(motion, cond)

                # book keeping per sample
                self._handle_log(epoch)
                self._handle_save()
                self.step += 1

            # --- end of the epoch ---
            if not self.t2m_dataset_ptr.is_generated_samples_already_on_disk(
                epoch, self.args_json_md5
            ):
                # generate a new batch of samples, randomly take conditioning information
                # from the dataset in memory and record what keys were sampled
                dataset_iterator = iter(self.data)

                # must iterate at least once
                num_iters_needed = -(
                    -self.samples_to_generate_per_epoch // self.batch_size
                )

                print(
                    f"To generate {self.samples_to_generate_per_epoch} samples this epoch with a batch size of {self.batch_size}, I must iterate {num_iters_needed} time(s)"
                )

                motions = []
                idxs_cat = []
                conds = []

                for j in range(num_iters_needed):
                    print(f"Sample generator iteration: {j}...")

                    # see : data_loaders/tensors.py
                    # s contains as many samples as the batch size
                    s: T_IDX_COLLATED_DIFFUSION_SAMPLE = next(dataset_iterator)

                    # the conditioning information for these samples and their keys (names) in
                    # the data dict, we omit their actual ground truth data here (which is the first
                    # dimension of the below 3 tuple):
                    _, model_kwargs, idxs = s

                    # generate samples and store them in 'raw' format. Raw format enables us to convert
                    # from the direct output of the diffusion model to one of the below:
                    # - format used for storing samples during training in memory
                    # - format used for visualizing already generated samples
                    # (
                    #   list of tensors (1 per rep) - each is a motion - shape: [bs, 263, 1, seq_len],
                    #   tensor of indexes used to generate samples from dataset: [bs, 1],
                    #   number of reps: int,
                    #   dictionary of conditioning information, stored in 'y'. Getting the value at key 'y' gets a DiffusionConditioningDict
                    # )
                    raw_sample: T_RAW_SAMPLE_TUPLE = self.generate_raw_samples(
                        model_kwargs, idxs, num_reps=1
                    )

                    (
                        motion_tensor_list,
                        idx_of_conditioning,
                        num_reps,
                        conditioning_info,
                    ) = raw_sample

                    # fix generation here to 1 rep, so get the 1st element in the list
                    motions.append(motion_tensor_list[0])
                    idxs_cat.append(idx_of_conditioning)
                    conds.append(conditioning_info["y"])

                # combine all the info
                k = self.samples_to_generate_per_epoch
                motions = torch.cat(motions)[:k,]
                idxs = torch.cat(idxs_cat)[:k,]

                # NB: no support for action, action_text
                mask = torch.cat([x["mask"] for x in conds[:k]])
                lengths = torch.cat([x["lengths"] for x in conds[:k]])
                text = functools.reduce(
                    operator.iconcat, [x["text"] for x in conds[:k]], []
                )
                tokens = functools.reduce(
                    operator.iconcat, [x["tokens"] for x in conds[:k]], []
                )

                raw_sample = (
                    [motions],
                    idxs,
                    num_reps,
                    {
                        "y": {
                            "mask": mask,
                            "lengths": lengths,
                            "text": text,
                            "tokens": tokens,
                            "action": None,
                            "action_text": [],
                        }
                    },
                )

                self.t2m_dataset_ptr.save_generated_samples_to_disk(
                    {"raw_samples": raw_sample}, epoch, self.args_json_md5
                )

                # encourage gc to free these objects
                del raw_sample
                del mask
                del lengths
                del text
                del tokens
                del motions
                del idxs_cat
                del conds
                del idxs
                del model_kwargs
                del motion_tensor_list
                del idx_of_conditioning
                del num_reps
                del conditioning_info

                # free memory
                gc.collect()
            else:
                print(f"Generated samples for epoch {epoch} already exist... skipping")

            if self._should_stop_curr_epoch():
                break

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()
