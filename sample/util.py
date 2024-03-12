"""Utilities for generating samples from an already trained MDM model with known configuration.
"""
from typing import List
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from torch.utils.data import DataLoader  # noqa

from train.train_platforms import NoPlatform
from train.training_loop import SelfConsumingTrainLoop

from model.cfg_sampler import ClassifierFreeSampleModel

from sample.types import T_IDX_COLLATED_DIFFUSION_SAMPLE, Prompts

from data_loaders.tensors import collate
from data_loaders.get_data import get_dataset_loader

from utils import dist_util
from utils.parser_util import (
    add_base_options,
    add_evaluation_options,
    get_cond_mode,
    overwrite_args_from_argparser,
    add_data_options,
    add_model_options,
    add_diffusion_options,
)
from utils.model_util import create_model_and_diffusion, load_model_wo_clip

MAX_MOTION_LENGTH_IN_FRAMES: int = SelfConsumingTrainLoop.HML_MAX_FRAMES
MIN_MOTION_LENGTH_IN_FRAMES: int = 1


def _dynamic_model_loader_argparser(model_path: Path) -> Namespace:
    # this code from utils.parser_util
    parser = ArgumentParser()

    # add options to capture defaults
    add_base_options(parser)
    add_evaluation_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)

    # parse them
    args, _ = parser.parse_known_args()

    # manually set the model path and configure
    args.model_path = str(model_path)
    args = overwrite_args_from_argparser(args, parser)
    cond_mode = get_cond_mode(args)
    args.cond_mode = cond_mode

    return args


def load_mdm_model_from_model_path_for_sampling(
        model_folder: Path, checkpoint_filename: str = "model.pt"
) -> SelfConsumingTrainLoop:
    """Given an enclosing folder for a model, load and prepare it for sampling.

    Args:
        model_folder: the Path object to the location of the model folder on disk.
        checkpoint_filename: the name of the .pt file to load, by default is `model.pt`.

    The model_folder must contain at least these files:

        model_folder
            - model.pt
            - args.json

    This means before using the pretrained models from MDM, for instance, one must rename the checkpoint
    from `model000475000.pt` to `model.pt`. OR modify the checkpoint_filename parameter to match the
    desired checkpoint file to load.

    Returns:
        an instance of `SelfConsumingTrainLoop`, which is a wrapper that has convenience
        functions for sampling from the loaded model. The model itself is set to the .model
        property of the `SelfConsumingTrainLoop` object.
    """
    # locate the checkpoint file
    model_checkpoint_file = model_folder / checkpoint_filename

    # given the checkpoint file, use the MDM argparser to determine its cofiguration (expects there to
    # be an args.json file)
    args = _dynamic_model_loader_argparser(model_checkpoint_file)

    # load dataset, but only the eval set - technically not needed because we are just going to sample
    data: DataLoader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=MAX_MOTION_LENGTH_IN_FRAMES,
        split="val",
        hml_mode="eval",
    )

    # load MDM
    model, diffusion = create_model_and_diffusion(args, data)
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        # wrapping model with the classifier-free sampler
        model = ClassifierFreeSampleModel(model)

    # disable random masking
    model.to(dist_util.dev())

    # not using this for training
    model.eval()

    # return wrapper around the model that can be used to sample from it
    return SelfConsumingTrainLoop(
        args, NoPlatform(), model, diffusion, data, util_mode=True
    )


def get_prompts_for_model_from_list_of_strings_and_lengths(
        prompts: Prompts,
        batch_size: int,
        placeholder_prompt: str = "The person stands still.",
) -> List[T_IDX_COLLATED_DIFFUSION_SAMPLE]:
    """Given a list of prompts, prepare them for use as conditioning information for an MDM model.

    Args:
        prompts: a list of tuples, describing ("prompt description", output duration in frames),
            e.g. [
                    ("a person walks to the right", 100),
                    ("a person picks up the box", 120),
                ]
        batch_size: the number of output motions to generate in a single batch. The prompts will
            be batched into groups of this size.
        placeholder_prompt: a prompt to use when the final batch is not equal to `batch_size` so that
            all batches are consistent. The default placeholder is: `The person stands still.`.

    Returns:
        a list of conditioning information to be given to an MDM model when sampling from it that will
        produce motions based on the given prompts and desired durations.
    """
    collate_args = []
    prompt_texts = []

    # this will be the minimum length of any motion in the list given
    min_motion_length = MAX_MOTION_LENGTH_IN_FRAMES

    for prompt_text, motion_length in prompts:
        if not (
                MIN_MOTION_LENGTH_IN_FRAMES <= motion_length <= MAX_MOTION_LENGTH_IN_FRAMES
        ):
            # oob
            raise ValueError(
                f"A motion sequence must have a frame count in the "
                f"interval: [{MIN_MOTION_LENGTH_IN_FRAMES}, {MAX_MOTION_LENGTH_IN_FRAMES}]"
            )

        min_motion_length = min(min_motion_length, motion_length)
        collate_args.append(
            {
                "inp": torch.zeros(motion_length),
                "tokens": None,
                "lengths": motion_length,
            }
        )
        prompt_texts.append(prompt_text)

    # pad out the end of the conditioning information to maintain consistent batch size
    to_fill = (batch_size - len(prompts)) % batch_size
    padding_args = to_fill * [
        {
            "inp": torch.zeros(min_motion_length),
            "tokens": None,
            "lengths": min_motion_length,
        }
    ]
    padding_texts = to_fill * [placeholder_prompt]

    collate_args.extend(padding_args)
    prompt_texts.extend(padding_texts)

    # format like how is done in sample/generate.py
    conditioning_infos = []
    for i in range(0, len(collate_args), batch_size):
        # b_ meaning batched
        b_collate_args = collate_args[i: i + batch_size]
        b_prompt_texts = prompt_texts[i: i + batch_size]

        b_collate_args = [
            dict(arg, text=txt) for arg, txt in zip(b_collate_args, b_prompt_texts)
        ]

        conditioning_info: T_IDX_COLLATED_DIFFUSION_SAMPLE = collate(b_collate_args)
        conditioning_infos.append(conditioning_info)

    return conditioning_infos
