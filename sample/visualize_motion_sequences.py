"""
Given a collection of model checkpoints of MDM, for each, sample k propts, with n samples each.
With those samples, produce a 3D rendered video of the sequence.
"""

# import json
from typing import List, Tuple, Dict, Any
from pathlib import Path
from functools import partial
import os
# from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from torch.utils.data import DataLoader

# from model.cfg_sampler import ClassifierFreeSampleModel

from sample.types import T_IDX_COLLATED_DIFFUSION_SAMPLE, T_RAW_SAMPLE_TUPLE

# from train.train_platforms import NoPlatform
# from train.training_loop import SelfConsumingTrainLoop

from utils import dist_util
# from utils.parser_util import (
#     add_base_options,
#     add_evaluation_options,
#     parse_and_load_from_model,
#     get_cond_mode,
#     overwrite_args_from_argparser,
#     add_data_options,
#     add_model_options,
#     add_diffusion_options,
# )
# from utils.model_util import create_model_and_diffusion, load_model_wo_clip

from data_loaders.get_data import get_dataset_loader
from data_loaders.tensors import collate
from data_loaders.humanml.data.dataset import Text2MotionDatasetV2

# from sample.generate import (
#     process_and_set_args_for_generate,
#     loop_generate,
#     load_dataset,
# )

from translation.amass_to_mdm import recover_from_ric
from translation.mdm_to_amass import mdm_to_skin_result, VposerSkinResult

from visualize.pro_render import skin_result_to_video
import sys
from time import sleep


# sRGBA color
ORANGE = [1.0, 0.35, 0.0, 1.0]  # A
BLUE = [0.0, 0.7, 1.0, 1.0]  # E
GREEN = [0.5, 1.0, 0.0, 1.0]  # D
GRAY = [0.55, 0.55, 0.55, 1.0]  # GROUND TRUTH

EXP_COLOR = {
    "ground_truth" : GRAY,
    "baseline": ORANGE,
    "iterative_finetuning": GREEN,
    "iterative_finetuning_with_correction": BLUE,
}


def single_motion_to_skin_result(
        single_motion_sequence: np.ndarray, num_joints=22
) -> VposerSkinResult:
    # turn (seq_len, 263) into --> (1, seq_len, 263), and make it a tensor
    motion_formatted = torch.from_numpy(single_motion_sequence).unsqueeze(0).float()

    # recover dimension: (1, seq_len, num_joints, 3)
    # recover_from_ric expects data shape: (1, seq_len, 263)
    pos_data = recover_from_ric(motion_formatted, num_joints).cpu().numpy()

    # pass to skin function as: (1, num_joints, 3, seq_len)
    pos_data = pos_data.transpose(0, 2, 3, 1)

    # skin the result, motions lengths is just [seq_len]
    skin_results = mdm_to_skin_result(pos_data, np.array([pos_data.shape[-1]]))

    # return the only skin result
    return skin_results[0]


def skin_result_to_multiple_videos(
        outputs_dir: Path, base_name: str, skin_result: VposerSkinResult, person_color=None
) -> None:
    # tuples of (camera translation, output filename)
    video_configs = [
        # video filenames are '/009871_${base_name}_view_${a|b}.gif'
        ([0.0, 1.0, 5.0], outputs_dir / (base_name + "_view_a")),
        ([0.0, 0.0, 5.0], outputs_dir / (base_name + "_view_b")),
    ]

    # render n different views of the same motion
    super_suffix = "_super.gif"
    normal_suffix = ".gif"
    for video_config in video_configs:
        _cam, _fname = video_config
        _fname: Path

        print("Saving a video to location: {0}".format(str(_fname)))

        p = Path(str(_fname) + super_suffix)
        if p.is_file():
            # prioritze the super render, overwrite the non-super one
            print(f"A video at {str(p)} already exists! Skipping.")
            continue

        # super
        skin_result_to_video(
            skin_result,
            output_path=str(_fname) + super_suffix,
            # dimensions of the video, (height, width)
            h=1600,
            w=2400,
            # --- the below args are optional, if left as `None`, function uses defaults ---
            #
            # camera_translation is the x, y, z position of the camera. The default value is: (0.0, 2.0, 10.0)
            #
            # here are two presets values for the camera position that seem to work well:
            #   completely level: [0.0, 0.0, 5.0]
            #       the floor plane is a line on the screen, and we can see more clearly float, penetrate, skate physics metrics
            #   visually appealing: [0.0, 1.0, 5.0]
            #       reasonably decent camera angle that showcases an overhead view of someone walking around the floor plane
            camera_translation=_cam,
            # the default color is gray
            person_color_rgb_perc=person_color or GRAY,
            # the default floor color is indigo
            floor_color_rgb_perc=None,
            with_joint_and_snapshots=True,
            snapshot_interval=40,
        )

        # regular
        skin_result_to_video(
            skin_result,
            output_path=str(_fname) + normal_suffix,
            h=800,
            w=1200,
            camera_translation=_cam,
            person_color_rgb_perc=person_color or GRAY,
            floor_color_rgb_perc=None,
            with_joint_and_snapshots=False,
        )

        # warning: if an exception is thrown somewhere in this block, the output video file may get corrupted


# from checkpoint_visual_sampler import single_motion_to_skin_result
# from checkpoint_visual_sampler import skin_result_to_multiple_videos


if __name__ == "__main__":
    """
    Go to the root directory of this repo: /motion-diffusion-model and run:

        python -m sample.checkpoint_visual_sampler --checkpoint_visual_folder ...

    If you see this:

        AttributeError: 'NoneType' object has no attribute 'glGetError'

    You may need to do the following in the shell:

        module load mesa patchelf glew
        export PYOPENGL_PLATFORM=osmesa
        export PYTHONPATH=$pwd

    'osmesa' doesn't seem to anti-alias the render. Try removing the `PYOPENGL_PLATFORM` variable
    from the environment to default to pyglet. Pyglet appears to anti-alias the render by default.

    Expects inputs to be as follows:

        See visualize/to_visualize

        Contains directories:
        - checkpoints
        - motions_gt
        - texts

        Example:
        - checkpoints/40k_5k_25_generation_16_latest_exp_A.pt
        - motions_gt/009871.npy
        - texts/009871.txt
    """
    # run this from root mdm dir
    motions_path = Path(str(sys.argv[1])) / "synthetic_motions"
    motions_visualized_path = Path(str(sys.argv[1])) / "synthetic_motions_viz"

    # TO_VISUALIZE_PATH = Path("exp_outputs/dataset_0064/visualization") 
    # VIDEOS_DIR_NAME = "videos"
    # TEXT_DIR = TO_VISUALIZE_PATH / "texts"

    """
    # get the directories containing checkpoint files
    all_checkpoint_dirs = [
        x for x in TO_VISUALIZE_PATH.glob("checkpoints/*") if x.is_dir()
    ]


    checkpoint_dirs = []
    for c in all_checkpoint_dirs:
        sleep(0.5)

        p_in_progress = c / "has_job.txt"
        if p_in_progress.is_file():
            print(
                f"{p_in_progress} already had a file there. Something else must be working on it. Skipping"
            )
            continue

        # only do one
        p_in_progress.write_text("started")
        checkpoint_dirs.append(c)
        break
    # checkpoint_dirs = all_checkpoint_dirs
    """

    # motions and texts have the same names, sort them for stability (hopefully the filenames are unique)
    motion_paths = sorted(os.listdir(motions_path))

    # --- LOAD GROUND TRUTHS AND RENDER THEM ---
    motions_visualized_path.mkdir(exist_ok=True, parents=True)
    for motion_filename in motion_paths[:20]:

        import pdb; pdb.set_trace()

        print(f"Processing ground truth motion sequence saved at: {motion_filename}")
        motion_path = os.path.join(motions_path, motion_filename)
        single_motion_sequence = np.load(str(motion_path)) # (71, 263)
        skin_result = single_motion_to_skin_result(single_motion_sequence)

        # e.g. 009871
        base_name = motion_filename.split(".")[0]

        # produce n videos per skin result
        skin_result_to_multiple_videos(motions_visualized_path, base_name, skin_result)

    quit()


    # --- PREPARE TO SAMPLE MDM ---

    # get configuration
    max_frames = 196
    motion_framerate = 20.0
    sampler_batch_size = 64
    # samples_per_prompt = int(sys.argv[2])

    # --- CREATE PROMPTS FOR SAMPLING ---
    # get prompt information from text files
    f_process_lines = partial(
        caption_to_motion_info, max_frames=max_frames, motion_framerate=motion_framerate
    )
    collate_args = []
    prompt_texts = []
    caption_filenames = []
    output_information = []
    for text_filename in text_filenames:
        base_name = text_filename.replace(".txt", "")

        # read the lines in
        raw_lines = Text2MotionDatasetV2.get_codec_lines(TEXT_DIR, base_name)

        # each element is a tuple of (caption, tokens, motion_len)
        lines = list(map(f_process_lines, raw_lines))

        for prompt_id, line in enumerate(lines):
            caption, tokens, motion_length = line

            for sample_num in range(samples_per_prompt):
                collate_args.append(
                    {
                        "inp": torch.zeros(motion_length),
                        "tokens": None,
                        "lengths": motion_length,
                    }
                )
                prompt_texts.append(caption)

                # experiment_ID_{A or D or E}_motion_ID_{motion_ID}_text_ID_{text_prompt_ID}_sample_ID_{sample_ID}_imitated_view_{a or b}.gif
                # experiment_ID_{A or D or E}_motion_ID_{motion_ID}_text_ID_{text_prompt_ID}_sample_ID_{sample_ID}_motion.npy
                output_information.append(
                    {
                        "motion_filename": text_filename,
                        # the actual text that was used to generate the prompt
                        "text_prompt_contents": line,
                        # the number of frames we want the motion to be
                        "motion_length": motion_length,
                        # each text file may contain several prompts, the ID is the index in which they appear in the promtp file
                        "text_prompt_id": prompt_id,
                        # the sample id is the index of the sample. If we want 4 samples per prompt, this is between [0-3].
                        "sample_id": sample_num,
                        # motion_id is the source filename of the prompt, e.g. M012005
                        "motion_id": base_name,
                        # not known at this time yet
                        "experiment_id": None,
                    }
                )

    # ensure conditioning info is as large as batch size
    num_samples = len(prompt_texts)
    to_fill = (sampler_batch_size - len(prompt_texts)) % sampler_batch_size
    padding_args = [
                       {"inp": torch.zeros(motion_length), "tokens": None, "lengths": motion_length}
                   ] * to_fill
    padding_texts = [
                        "someoone walks to the right."
                    ] * to_fill  # just a placeholder prompt
    collate_args.extend(padding_args)
    prompt_texts.extend(padding_texts)

    # format like how is done in sample/generate.py
    conditioning_infos = []
    for i in range(0, len(collate_args), sampler_batch_size):
        b_collate_args = collate_args[i: i + sampler_batch_size]
        b_prompt_texts = prompt_texts[i: i + sampler_batch_size]

        b_collate_args = [
            dict(arg, text=txt) for arg, txt in zip(b_collate_args, b_prompt_texts)
        ]
        conditioning_info: T_IDX_COLLATED_DIFFUSION_SAMPLE = collate(b_collate_args)
        conditioning_infos.append(conditioning_info)

    # we don't really need this but our code expects it to be non-null. Load the smallest split of the dataset
    # for speed
    dataset_name = "humanml"
    data: DataLoader = get_dataset_loader(
        name=dataset_name,
        batch_size=32,
        num_frames=max_frames,
        split="val",
        hml_mode="eval",
    )

    # --- RENDER THE VIDEOS OF EACH PROMPT FOR EACH MODEL ---
    for model_path in checkpoint_dirs:
        # each model gets sample prompt information
        render_videos_for_model_checkpoint(
            TO_VISUALIZE_PATH,
            VIDEOS_DIR_NAME,
            model_path,
            output_information,
            num_samples,
            conditioning_infos,
            sampler_batch_size=sampler_batch_size,
            caption_filenames=caption_filenames,
        )
