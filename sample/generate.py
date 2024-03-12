"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.

This code is based on https://github.com/openai/guided-diffusion
"""
import functools
import operator
import os
import shutil
from argparse import Namespace
from os.path import join as pjoin
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.data.dataset import Text2MotionDatasetV2, load_pickled_np_dict
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate
from model.cfg_sampler import ClassifierFreeSampleModel
from sample.types import (
    T_HUMANML3D_KIT_DATASET_MODE,
    T_HUMANML3D_KIT_DATASET_SPLIT_TYPE,
    T_IDX_COLLATED_DIFFUSION_SAMPLE,
    T_RAW_SAMPLE_TUPLE,
    T_SAMPLE_RETURN_FORMAT,
    GeneratedSampleBatchDict,
)
from train.train_platforms import NoPlatform
# from train.training_loop import SelfConsumingTrainLoop
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import generate_args
from visualize.generated_sample_to_mp4 import visualize_samples_and_save_to_disk

from tqdm import tqdm
import time
sep_str = "------------------------------------------------------------------------------------"


def process_and_set_args_for_generate(
    args: Namespace, enforce_batch_size: bool = True
) -> Tuple:
    """
    Warning: this mutates the input args object.
    """
    out_path: str = args.output_dir
    name: str = os.path.basename(os.path.dirname(args.model_path))
    niter: str = (
        os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    )

    # the maximum total frames we can generate
    # --motion_length defaults to 6.0 seconds (see parser_util.py)
    # The max for HumanML3D is 9.8 (text-to-motion)
    # The max for HumanAct12 is 2.0 (action-to-motion)
    max_frames: int = 196 if args.dataset in ["kit", "humanml"] else 60
    fps: float = 12.5 if args.dataset == "kit" else 20
    n_frames = min(max_frames, int(args.motion_length * fps))

    is_using_data = not any(
        [args.input_text, args.text_prompt, args.action_file, args.action_name]
    )

    if not out_path:
        out_path = os.path.join(
            os.path.dirname(args.model_path),
            "samples_{}_{}_seed{}".format(name, niter, args.seed),
        )
        if args.text_prompt != "":
            out_path += "_" + args.text_prompt.replace(" ", "_").replace(".", "")
        elif args.input_text != "":
            out_path += "_" + os.path.basename(args.input_text).replace(
                ".txt", ""
            ).replace(" ", "_").replace(".", "")

    # this block must be called BEFORE the dataset is loaded
    action_text: List[str] = []
    texts: List[str] = []
    if args.text_prompt != "":
        # use the given text prompt - it counts as a single sample
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != "":
        # input text is a path to a text file that contains a prompt on each
        # line
        assert os.path.exists(args.input_text)
        with open(args.input_text, "r") as fr:
            texts = fr.readlines()
        texts = [s.replace("\n", "") for s in texts]
        args.num_samples = len(texts)

        print("Loaded text prompts from file. Prompts:")
        print(texts)
    elif args.action_name:
        # human act 12
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != "":
        # human act 12 with a text file that defines an action per line
        assert os.path.exists(args.action_file)
        with open(args.action_file, "r") as fr:
            action_text = fr.readlines()
        action_text = [s.replace("\n", "") for s in action_text]
        args.num_samples = len(action_text)

    if enforce_batch_size:
        # So why do we need this check? In order to protect GPU from a memory overload in the following line.
        # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
        # If it doesn't, and you still want to sample more prompts, run this script with different seeds
        # (specify through the --seed flag)
        assert (
            args.num_samples <= args.batch_size
        ), f"Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})"

        # Sampling a single batch from the testset, with exactly args.num_samples
        args.batch_size = args.num_samples

    return args, max_frames, n_frames, is_using_data, texts, action_text, out_path, fps


def main() -> None:
    # 1-off generation for experiement 3a
    is_generate_for_3a = False

    # --- CONFIG ---
    args = generate_args()
    if is_generate_for_3a:
        print("*** EXPERIMENT 3a ***")
    else:
        fixseed(args.seed)

    # 1-off generation for POC samples
    is_generate_POC_samples = args.is_generate_POC_samples

    # ---- configure for experiment 3a ---
    split = "train" if is_generate_for_3a or is_generate_POC_samples else "test"
    return_format: T_SAMPLE_RETURN_FORMAT = (
        "in_memory"
        if is_generate_for_3a
        else "save_raw_motion"
        if is_generate_POC_samples
        else "to_visualize"
    )
    num_reps = (
        1 if is_generate_for_3a or is_generate_POC_samples else args.num_repetitions
    )
    hml_mode = "train" if is_generate_for_3a or is_generate_POC_samples else "text_only"
    enforce_batch_size = (
        False if is_generate_for_3a or is_generate_POC_samples else True
    )
    # ------------------------------------

    # parse the args into what we need to run this
    (
        args,
        max_frames,
        n_frames,
        is_using_data,
        texts,
        action_text,
        out_path,
        fps,
    ) = process_and_set_args_for_generate(args, enforce_batch_size)

    # configure distributed training settings
    dist_util.setup_dist(args.device)

    # --- MODEL LOADING ---
    print("Loading dataset...")

    total_num_samples = args.num_samples * num_reps

    data = load_dataset(args, max_frames, n_frames, split=split, hml_mode=hml_mode)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        # wrapping model with the classifier-free sampler
        model = ClassifierFreeSampleModel(model)

    model.to(dist_util.dev())
    # disable random masking
    model.eval()

    # --- DIFFUSION SAMPLING & FORMATTING ---
    training_util = SelfConsumingTrainLoop(
        args, NoPlatform(), model, diffusion, data, util_mode=True
    )
    training_util.sampler_batch_size = args.batch_size

    # --- DATA LOADING ---
    if is_using_data:
        iterator = iter(data)

        # determine how many samples we want to generate
        num_iters_needed = -(-total_num_samples // training_util.sampler_batch_size)

        print(
            f"To generate {total_num_samples} samples with a batch size of {training_util.sampler_batch_size}, "
            f"I must iterate {num_iters_needed} time(s)"
        )

        for generation in range(num_iters_needed):
            x: T_IDX_COLLATED_DIFFUSION_SAMPLE = next(iterator)
            generate_and_visualize_samples(
                args,
                num_reps,
                training_util,
                x,
                out_path,
                fps,
                generation=generation,
                return_format=return_format,
                # generating many samples, do not render them as video
                render_video=False,
            )
    else:
        collate_args = [
            {"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}
        ] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            # add text as a key in the list of dictionaries for args
            collate_args = [
                dict(arg, text=txt) for arg, txt in zip(collate_args, texts)
            ]
        else:
            # a2m
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [
                dict(arg, action=one_action, action_text=one_action_text)
                for arg, one_action, one_action_text in zip(
                    collate_args, action, action_text
                )
            ]

        # process batch, model_kwargs is the conditioning information for
        # diffusion
        x: T_IDX_COLLATED_DIFFUSION_SAMPLE = collate(collate_args)
        generate_and_visualize_samples(
            args,
            num_reps,
            training_util,
            x,
            out_path,
            fps,
            generation=0,
            return_format=return_format,
        )


def generate_and_visualize_samples(
    args: Namespace,
    num_reps: int,
    training_util,#: SelfConsumingTrainLoop,
    x: T_IDX_COLLATED_DIFFUSION_SAMPLE,
    out_path: str,
    fps: int,
    generation: int = 0,
    return_format: T_SAMPLE_RETURN_FORMAT = "to_visualize",
    render_video: bool = True,
    delete_if_exists: bool = True,
) -> None:
    _, model_kwargs, idxs = x

    # result: (263,)-dimensional if 22 joints
    raw_sample: T_RAW_SAMPLE_TUPLE = training_util.generate_raw_samples(
        model_kwargs,
        idxs,
        num_reps=num_reps,
    )
    # --- SAVE SAMPLES TO DISK ---
    if delete_if_exists:
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path)

    npy_path = os.path.join(out_path, f"results_{generation}.npy")

    # --- VISUALIZE SAMPLES AS VIDEOS AND SAVE TO DISK ---
    if return_format == "to_visualize":
        # result: (22, 3)-dimensional if 22 joints
        all_motions, all_lengths, all_caption, _ = training_util.process_raw_samples(
            raw_sample,
            return_format=return_format,
        )

        # save to disk in an object like this:
        samples: GeneratedSampleBatchDict = {
            "motion": all_motions,
            "text": all_caption,
            "lengths": all_lengths,
            "num_samples": args.num_samples,  # idxs.shape[0],
            "num_repetitions": num_reps,
            "conditioning_idxs": idxs,
        }

        # motions shape is (3, 24, 3, 196)
        # save all results
        print(f"saving results file to [{npy_path}]")
        np.save(npy_path, samples)

        with open(npy_path.replace(".npy", ".txt"), "w") as fw:
            # write the texts
            fw.write("\n".join(all_caption))

        with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
            # write the lengths
            fw.write("\n".join([str(l) for l in all_lengths]))

        if render_video:
            visualize_samples_and_save_to_disk(
                args.dataset,
                args.unconstrained,
                samples,
                out_path,
                fps,
                num_samples=args.num_samples,
                num_reps=num_reps,
                batch_size=args.batch_size,
            )
    elif return_format == "save_raw_motion":
        # result: (22, 3)-dimensional if 22 joints
        (
            all_motions,
            all_lengths,
            all_caption,
            all_tokens,
        ) = training_util.process_raw_samples(
            raw_sample,
            return_format=return_format,
        )

        # return the data that was generated by the model
        return all_motions, all_lengths, all_caption, all_tokens
    elif return_format == "save_generated_motion":
        # minimal processing, but not save to .tmp folder
        (
            all_motions,
            all_lengths,
            all_caption,
            all_tokens,
        ) = training_util.process_raw_samples(
            raw_sample,
            return_format="in_memory",
        )
        return all_motions, all_lengths, all_caption, all_tokens
    else:
        motions = []
        idxs_cat = []
        conds = []

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
        k = idxs.shape[0]

        motions = torch.cat(motions)[:k,]
        idxs = torch.cat(idxs_cat)[:k,]

        # NB: no support for action, action_text
        mask = torch.cat([x["mask"] for x in conds[:k]])
        lengths = torch.cat([x["lengths"] for x in conds[:k]])
        text = functools.reduce(operator.iconcat, [x["text"] for x in conds[:k]], [])
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

        np.save(npy_path, {"raw_samples": raw_sample})

        print(f"Samples were saved in the 'to memory' format to {npy_path}.")

        if not render_video:
            # ignore rendering to a video
            return

        raw_sample_tuple: T_RAW_SAMPLE_TUPLE = load_pickled_np_dict(npy_path)[
            "raw_samples"
        ]
        print(f"Reloaded raw samples from: {npy_path}")

        all_motions, all_lengths, all_caption, _ = training_util.process_raw_samples(
            raw_sample_tuple,
            return_format="to_visualize",
        )
        # prepare for visualize
        samples: GeneratedSampleBatchDict = {
            "motion": all_motions,
            "text": all_caption,
            "lengths": all_lengths,
            "num_samples": idxs.shape[0],
            "num_repetitions": num_reps,
            "conditioning_idxs": idxs,
        }

        print("Visualizing the previously in memory samples...")
        visualize_samples_and_save_to_disk(
            args.dataset,
            args.unconstrained,
            samples,
            out_path,
            fps,
            num_samples=args.num_samples,
            num_reps=num_reps,
            batch_size=args.batch_size,
        )


def load_dataset(
    args: Namespace,
    max_frames: int,
    n_frames: int,
    split: T_HUMANML3D_KIT_DATASET_SPLIT_TYPE = "test",
    hml_mode: T_HUMANML3D_KIT_DATASET_MODE = "text_only",
) -> DataLoader:
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split=split,
        hml_mode=hml_mode,
    )
    if args.dataset in ["kit", "humanml"]:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


def loop_generate(
    prompts: List[Dict[str, str]],
    model,
    diffusion,
    data,
    model_path_: Optional[str] = None,
    output_dir_: Optional[str] = None,
    load_motion_to_get_length: bool = True,
) -> None:
    """
    Provide a list of prompts, formatted as a dictionary {"text": "your prompt", "output": "save npy file here.npy"}, e.g.:
    [
        {"text": "someone is picking something up off the ground", "file": "1241951.npy"},
        {"text": "a person is walking backwards, with their hands in their pockets", "file": "ARQ_12419qwrq.npy"},
        ...
    ]

    The files should be the name of the output, but not including the enclosing directory to which they are saved. So in the
    above, specifying the `output_dir_` argument to be: "./save/generation_10" would mean the above prompts will be saved to:
        `./save/generation_10/1241951.npy`
        `./save/generation_10/ARQ_12419qwrq.npy`
    respectively.

    Both `save_samples_to_` and `model_path_` args can be set explicitly in this function arguments or given as command line arguments.
    The CLI arguments are:
        --model_path ... (`model_path_`)
        --output_dir ... (`output_dir_`)

    Explictly setting them in the function arguments takes precedence over the CLI arguments.
    """
    VISUALIZE_FOR_TESTING = False
    data_root = "dataset/HumanML3D"
    motion_dir = pjoin(data_root, "new_joint_vecs")
    # three inputs
    # 1. model path
    # 2. dictionary containing the prompts, each like: {'text': "", 'output': ""}
    # 3. output directory

    # --- CONFIG ---
    args = generate_args()
    args.model_path = model_path_ or args.model_path
    # fixseed(args.seed)
    # model_path = model_path_ or args.model_path
    output_dir = output_dir_ or args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    split = "train"
    return_format: T_SAMPLE_RETURN_FORMAT = "save_generated_motion"
    num_reps = 1
    hml_mode = "train"
    enforce_batch_size = False

    render_video = False
    if VISUALIZE_FOR_TESTING:
        # if this is true, script will save videos for the generated motions
        return_format = "to_visualize"
        render_video = True

    # ------------------------------------

    # parse the args into what we need to run this
    (
        args,
        max_frames,
        n_frames,
        _,
        # in original implementation, texts would be parsed from a text file
        _,
        _,
        out_path,
        fps,
    ) = process_and_set_args_for_generate(args, enforce_batch_size)

    """
    # --- MODEL LOADING ---
    print("Loading dataset...")
    data: DataLoader = load_dataset(
        args, max_frames, n_frames, split=split, hml_mode=hml_mode
    )

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)


    if args.guidance_param != 1:
        # wrapping model with the classifier-free sampler
        model = ClassifierFreeSampleModel(model)
    """        

    # disable random masking
    # model.to(dist_util.dev())
    model.eval()

    

    # --- DIFFUSION SAMPLING & FORMATTING ---
    from train.training_loop import SelfConsumingTrainLoop
    training_util = SelfConsumingTrainLoop(
        args, NoPlatform(), model, diffusion, data, util_mode=True
    )
    # batch_size = args.batch_size
    # batch_size = 512
    batch_size = 64 # current subset has 2775, so 64 is more appropriate for lower synth augmentation percents
    training_util.sampler_batch_size = batch_size

    total_num_samples = len(prompts)

    # --- DATA LOADING ---
    # n_frames is bound by max_frames (196)
    # and is set by the desired length is seconds passed to the loader arguments.
    # if unset, it has motion_length=6.0, where framerate = 20 so that is 120 for HML3D.
    collate_args = []
    for p in tqdm(prompts):
        text, filename = p["text"], p["file"]

        if load_motion_to_get_length:
            original_file = filename.split("-sampled.npy")[0] + ".npy"

            # in dataset.py -> Text2MotionDatasetV2, the bounds of a sequence
            # used in HumanML3D training are from in [40, 200] frames.
            motion = np.load(pjoin(motion_dir, original_file))

            # original model can generate at max 196 frame sequences
            num_frames = np.clip(len(motion), 40, 196)

            # set it in the original object
            p["length"] = num_frames
        else:
            # testing mode, take the length from the original prompts object
            num_frames = p["length"]

        collate_args.append(
            {
                "text": text,
                "tokens": None,
                "lengths": num_frames,
                "inp": torch.zeros(num_frames),
            }
        )

    for k in range(0, total_num_samples, batch_size):

        start = time.time()

        filenames = [x["file"] for x in prompts[k : k + batch_size]]
        lengths = [x["length"] for x in prompts[k : k + batch_size]]
        collated_texts = collate_args[k : k + batch_size]

        if len(filenames) < batch_size:
            # pad out the end with repeat / blank data
            to_add = batch_size - len(filenames)
            filenames += [None] * to_add
            lengths += [None] * to_add
            collated_texts += [collated_texts[-1]] * to_add

        x: T_IDX_COLLATED_DIFFUSION_SAMPLE = collate(collated_texts)

        all_motions, _, _, _ = generate_and_visualize_samples(
            args,
            num_reps,
            training_util,
            x,
            out_path,
            fps,
            generation=0,
            return_format=return_format,
            render_video=render_video,
            delete_if_exists=False,
        )

        for i in range(all_motions.shape[0]):
            filename = filenames[i]
            seq_len = lengths[i]
            if filename is None:
                # skip items added just to reach the batch size.
                continue
            if not filename.endswith(".npy"):
                filename += ".npy"

            # (196, 263)
            # crop it to the conditioning length
            single_motion = all_motions[i, :seq_len, :]
            save_to = Path(output_dir) / (filename)

            print(
                f"Save motion to: {save_to}. Had shape: {single_motion.shape}. Sequence length: {seq_len}"
            )

            np.save(str(save_to), single_motion)

        duration = time.time() - start
        print(f"{sep_str}\nTIME FOR THIS LAST GENERATED BATCH: {duration}, TIME PER GENERATED MOTION: {duration/batch_size}\n{sep_str}")


    # del data
    del model
    del diffusion
    # del state_dict
    del training_util
    del collate_args
    torch.cuda.empty_cache()


if __name__ == "__main__":
    """
    Example generation command from list of text prompts:
        python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --input_text ./assets/example_text_prompts.txt --num_repetitions 1


    Example generation to save individual .npy files for batched generated motion for a dictionary of
    text prompts:

        python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000475000.pt --output_dir ./save/batch_generated_multiprompt_test --is_generate_batched_text_prompt_samples True

    Can use the above command to test this file alone with placeholder data.
    """
    args = generate_args()

    if args.is_generate_batched_text_prompt_samples:
        import random

        test_batches = 0

        # default batch size is 64
        # 64 * 10 prompts, with 1,000 diffusion steps takes ~1,233 seconds (incl. dataset / model checkpoint load time)
        # approx: 40 sec + 1.86 sec/prompt
        prompts = [
            {
                "text": "a person jumps only once.",
                "file": f"test_{i}.npy",
                # inclusive bounds
                "length": random.randint(20, 196),
            }
            # use a range that doesn't evenly fit into a batch size just for testing
            # purposes
            for i in range((64 * test_batches) + 10)
        ]
        loop_generate(prompts, load_motion_to_get_length=False)
    else:
        main()
