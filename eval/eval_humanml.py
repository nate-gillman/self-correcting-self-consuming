from collections import OrderedDict
from datetime import datetime
from io import TextIOWrapper
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.motion_loaders.model_motion_loaders import (
    get_mdm_loader,  # get_motion_loader
)
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.utils.utils import *
from diffusion import logger
from model.cfg_sampler import ClassifierFreeSampleModel
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import evaluation_parser

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.motion_loaders.comp_v6_model_dataset import CompMDMGeneratedDataset
from data_loaders.humanml.data.dataset import HumanML3D

from translation.mdm_to_amass import mdm_to_skin_result
from eval.physics import compute_physics_metrics_for_skin_result


torch.multiprocessing.set_sharing_strategy("file_system")


def unwrap_multicollate_batch(batch_obj: Tuple):
    return batch_obj
    # print(len(batch_obj))
    # if len(batch_obj) == 2:
    #     # batch, indexes of samples from dataset
    #     batch, _ = batch_obj
    #     return batch
    # else:
    #     # is already unwrapped, 7-tuple
    #     return batch_obj


def evaluate_matching_score(
    eval_wrapper: EvaluatorMDMWrapper,
    motion_loaders: Dict[str, DataLoader],
    file: TextIOWrapper,
):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print("========== Evaluating Matching Score ==========")
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        with torch.no_grad():
            for idx, batch_obj in enumerate(motion_loader):
                batch = unwrap_multicollate_batch(batch_obj)
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens,
                )
                dist_mat = euclidean_distance_matrix(
                    text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy()
                )
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f"---> [{motion_loader_name}] Matching Score: {matching_score:.4f}")
        print(
            f"---> [{motion_loader_name}] Matching Score: {matching_score:.4f}",
            file=file,
            flush=True,
        )

        line = f"---> [{motion_loader_name}] R_precision: "
        for i in range(len(R_precision)):
            line += "(top %d): %.4f " % (i + 1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print("========== Evaluating FID ==========")
    with torch.no_grad():
        for idx, batch_obj in enumerate(groundtruth_loader):
            batch = unwrap_multicollate_batch(batch_obj)
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions, m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f"---> [{model_name}] FID: {fid:.4f}")
        print(f"---> [{model_name}] FID: {fid:.4f}", file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times: int):
    eval_dict = OrderedDict({})
    print("========== Evaluating Diversity ==========")
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f"---> [{model_name}] Diversity: {diversity:.4f}")
        print(f"---> [{model_name}] Diversity: {diversity:.4f}", file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print("========== Evaluating MultiModality ==========")
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch_obj in enumerate(mm_motion_loader):
                # batch = unwrap_multicollate_batch(batch_obj)
                batch, _ = batch_obj
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch

                # see: data_loaders/humanml/networks/evaluator_wrapper.py
                motion_embedings = eval_wrapper.get_motion_embeddings(
                    motions[0], m_lens[0]
                )
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f"---> [{model_name}] Multimodality: {multimodality:.4f}")
        print(
            f"---> [{model_name}] Multimodality: {multimodality:.4f}",
            file=file,
            flush=True,
        )
        eval_dict[model_name] = multimodality
    return eval_dict


def evaluate_physics(
    eval_wrapper: EvaluatorMDMWrapper,
    motion_loaders: Dict[str, DataLoader],
    file: TextIOWrapper,
):
    """Produce Physics-related Metrics: Float, Penetrate, Skate, Phys-Err"""
    print("========== Evaluating Physics ==========")
    eval_dict = OrderedDict({})
    import collections

    for motion_loader_name, motion_loader in motion_loaders.items():
        # produce physics metrics for this motion loader 
        floats, penetrates, skates, phys_err = [], [], [], []
        eval_dict[motion_loader_name] = collections.defaultdict(list)

        with torch.no_grad():
            for idx, batch_obj in enumerate(motion_loader):
                mdm_sampled_motions = []
                # extract the XYZ positions of joints from the 263 dim
                # hml3d vector
                batch = unwrap_multicollate_batch(batch_obj)

                # motions: (bs, seq_len, 263)
                _, _, _, _, motions, m_lens, _ = batch

                # undo what MDM dataloader does, get dim: (bs, 263, 1, seq_len)
                sample = motions.permute(0, 2, 1).unsqueeze(2)

                n_joints = 22 if sample.shape[1] == 263 else 21

                ds = motion_loader.dataset

                if isinstance(ds, CompMDMGeneratedDataset):
                    # dataset from MDM outputs
                    # undo the t2m evaluator norm step
                    sample = ((sample.permute(0, 2, 3, 1) * ds.dataset.std_for_eval) + ds.dataset.mean_for_eval).cpu().float()
                elif isinstance(ds, HumanML3D):
                    # ground truth, inv transform it
                    sample = motion_loader.dataset.t2m_dataset.inv_transform(
                        sample.cpu().permute(0, 2, 3, 1)
                    ).float()

                # here, sample.shape = torch.Size([1, 1, seq_len, n_joints, 3])
                sample = recover_from_ric(sample, n_joints)

                # here, sample.shape = torch.Size([1, n_joints, 3, seq_len])
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

                # ensure shape is (bs, seq_len, n_joints, 3)
                sample = sample.permute(0, 3, 1, 2)
                
                # at this point, sample: (bs, n_joints, 3, seq_len)
                mdm_sampled_motions.append(sample)

                # gather all the motions together from the dataloader
                # prepare so that dimension is (bs, n_joints, 3, seq_len)
                batched_motions = torch.cat(mdm_sampled_motions, dim=0).permute(0, 2, 3, 1).cpu().numpy()

                # compute the physics metrics for each sequence sampled from trained MDM
                # procedure is roughly:
                #   1. convert MDM to AMASS using VPoser
                #   2. skin the VPoser result to get the vertices
                #   3. given the geometry of the skin, compute physics metrics for single sequence
                skin_results = mdm_to_skin_result(batched_motions, m_lens.numpy())

                for skin_result in skin_results:
                    metrics = compute_physics_metrics_for_skin_result(skin_result)

                    # consolidate all
                    floats.append(metrics.float)
                    penetrates.append(metrics.penetrate)
                    skates.append(metrics.skate)
                    phys_err.append(metrics.phys_err)

                # try to free memory
                torch.cuda.empty_cache()
                del skin_results
                del batched_motions

            
        # gather all metrics from all batches and associate it with its motion loader
        eval_dict[motion_loader_name]["float"] += floats
        eval_dict[motion_loader_name]["penetrate"] += penetrates
        eval_dict[motion_loader_name]["skate"] += skates
        eval_dict[motion_loader_name]["phys_err"] += phys_err

        m = eval_dict[motion_loader_name]

        # print out the the results to stdout / file
        print(f"---> [{motion_loader_name}] Physics: {m}")
        print(
            f"---> [{motion_loader_name}] Physics: {m}",
            file=file,
            flush=True,
        )


    return eval_dict


def get_metric_statistics(
    values: np.ndarray, replication_times: int
) -> Tuple[np.float32, np.float32]:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(
    eval_wrapper: EvaluatorMDMWrapper,
    gt_loader: DataLoader,
    eval_motion_loaders: Dict[str, Tuple[DataLoader, DataLoader]],
    log_file: str,
    replication_times: int,
    diversity_times: int,
    mm_num_times: int,
    run_mm: bool = False,
    EVALUATE_PHYSICS: bool = False
) -> Dict[str, np.float32]:
    """
    replication_times, diversity_times, mm_num_times all influence how much time it takes to run
    the eval script.
    """
    with open(log_file, "w") as f:
        all_metrics = OrderedDict(
            {
                "Matching Score": OrderedDict({}),
                "R_precision": OrderedDict({}),
                "FID": OrderedDict({}),
                "Diversity": OrderedDict({}),
                "MultiModality": OrderedDict({}),
            }
        )
        for replication in range(replication_times):
            motion_loaders: Dict[str, DataLoader] = {}
            mm_motion_loaders = {}
            motion_loaders["ground truth"] = gt_loader

            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                # getter returns a tuple of DataLoader
                motion_loader, mm_motion_loader = motion_loader_getter()

                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(
                f"==================== Replication {replication} ===================="
            )
            print(
                f"==================== Replication {replication} ====================",
                file=f,
                flush=True,
            )

            if EVALUATE_PHYSICS:
                print(f"Time: {datetime.now()}")
                print(f"Time: {datetime.now()}", file=f, flush=True)
                phys_metrics_dict = evaluate_physics(eval_wrapper, motion_loaders, f)

            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(
                eval_wrapper, motion_loaders, f
            )


            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                # MULTIMODALITY
                print(f"Time: {datetime.now()}")
                print(f"Time: {datetime.now()}", file=f, flush=True)
                mm_score_dict = evaluate_multimodality(
                    eval_wrapper, mm_motion_loaders, f, mm_num_times
                )

            print(f"!!! DONE !!!")
            print(f"!!! DONE !!!", file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics["Matching Score"]:
                    all_metrics["Matching Score"][key] = [item]
                else:
                    all_metrics["Matching Score"][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics["R_precision"]:
                    all_metrics["R_precision"][key] = [item]
                else:
                    all_metrics["R_precision"][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics["FID"]:
                    all_metrics["FID"][key] = [item]
                else:
                    all_metrics["FID"][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics["Diversity"]:
                    all_metrics["Diversity"][key] = [item]
                else:
                    all_metrics["Diversity"][key] += [item]

            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics["MultiModality"]:
                        all_metrics["MultiModality"][key] = [item]
                    else:
                        all_metrics["MultiModality"][key] += [item]

        mean_dict: Dict[str, np.float32] = {}
        for metric_name, metric_dict in all_metrics.items():
            print("========== %s Summary ==========" % metric_name)
            print("========== %s Summary ==========" % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(
                    np.array(values), replication_times
                )
                mean_dict[metric_name + "_" + model_name] = mean
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(
                        f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}"
                    )
                    print(
                        f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}",
                        file=f,
                        flush=True,
                    )
                elif isinstance(mean, np.ndarray):
                    line = f"---> [{model_name}]"
                    for i in range(len(mean)):
                        line += "(top %d) Mean: %.4f CInt: %.4f;" % (
                            i + 1,
                            mean[i],
                            conf_interval[i],
                        )
                    print(line)
                    print(line, file=f, flush=True)

        # return all the collected metrics
        return mean_dict


if __name__ == "__main__":
    args = evaluation_parser()

    if args.no_fixseed:
        print("NB: not fixing the random seed.")
    else:
        fixseed(args.seed)

    # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    args.batch_size = 32

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    log_file = os.path.join(
        os.path.dirname(args.model_path), "eval_humanml_{}_{}".format(name, niter)
    )

    if args.guidance_param != 1.0:
        log_file += f"_gscale{args.guidance_param}"
    log_file += f"_{args.eval_mode}"
    log_file += ".log"

    print(f"Will save to log file [{log_file}]")
    print(f"Eval mode [{args.eval_mode}]")

    if args.eval_mode == "debug":
        num_samples_limit = 1000  # None means no limit (eval over all dataset)
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        # if HumanML3D has ~22k samples, diversity is measured by a subset
        # that is roughly ~1.36% of training data. Original MDM paper fixes this
        # as 300
        diversity_times = 300
        replication_times = 5  # about 3 Hrs
    elif args.eval_mode == "wo_mm":
        # wo means 'without'
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 20  # about 12 Hrs
    elif args.eval_mode == "mm_short":
        # 5 runs of multimodality
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        diversity_times = 300
        replication_times = 5  # about 15 Hrs
    else:
        raise ValueError()

    dist_util.setup_dist(args.device)
    logger.configure()

    # --- DATALOADERS ---
    logger.log("creating data loader...")
    split = "test"
    # ground truth
    gt_loader: DataLoader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=None,
        split=split,
        # gt uses data_loaders.humanml.data.dataset.collate_fn
        hml_mode="gt",
        subset_by_keyword=args.filter_dataset_by_keyword,
    )
    gen_loader: DataLoader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=None,
        split=split,
        hml_mode="eval",
        subset_by_keyword=args.filter_dataset_by_keyword,
    )

    num_actions = gen_loader.dataset.num_actions

    if args.filter_dataset_by_keyword:
        # make the number of samples to generate no larger than the evaluation set
        # if we want to filter the ground truth
        num_gt = len(gt_loader.dataset)
        num_samples_limit = num_gt

        # take about ~5% subset to evaluate diversity if looking at a
        # filtered dataset
        diversity_times = int(num_gt * 0.05)

        # fewer steps for small tests, defaults to 1000
        args.diffusion_steps = 100

    # --- MODELS ----
    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, gen_loader)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        # wrapping model with the classifier-free sampler
        model = ClassifierFreeSampleModel(model)
        model.to(dist_util.dev())

    # disable random masking
    model.eval()

    # --- GENERATED SAMPLES ---
    eval_motion_loaders: Dict[str, Tuple[DataLoader, DataLoader]] = {
        #######################
        ## HumanML3D Dataset ##
        #######################
        # returns CompMDMGeneratedDataset
        "vald": lambda: get_mdm_loader(
            model,
            diffusion,
            args.batch_size,
            gen_loader,
            mm_num_samples,
            mm_num_repeats,
            gt_loader.dataset.opt.max_motion_length,
            num_samples_limit,
            args.guidance_param,
        )
    }

    # --- EVALUATION ROUTINE ---
    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())

    mean_dict = evaluation(
        eval_wrapper,
        gt_loader,
        eval_motion_loaders,
        log_file,
        replication_times,
        diversity_times,
        mm_num_times,
        run_mm=run_mm,
    )
    print(mean_dict)
