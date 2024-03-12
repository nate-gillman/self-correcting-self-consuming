"""
Given a path to a checkpoint, prompt, and duration, sample a motion
"""
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch

from sample.types import T_RAW_SAMPLE_TUPLE, T_IDX_COLLATED_DIFFUSION_SAMPLE, Prompts
from train.training_loop import SelfConsumingTrainLoop

from sample.util import (
    get_prompts_for_model_from_list_of_strings_and_lengths,
    load_mdm_model_from_model_path_for_sampling
)


def sample_motion(
        prompts: Prompts, model_util: SelfConsumingTrainLoop, move_raw_to_cpu: bool = True,
) -> Tuple[np.ndarray, Union[np.ndarray, torch.Tensor]]:
    """Given an MDM model, sample it using specific prompts.

    One can load the MDM model from a specific file with the function:

        sample.util.load_mdm_model_from_model_path_for_sampling

    prompts:
        these must be a list of tuples of:
            ("the prompt describing the motion to generate", <int> number of frames for the result)

        for example, the prompt:
            ("a person walks to their right slowly", 120)

        will generate a motion of 120 frames, conditioned on the given caption.

    if move_raw_to_cpu is on, then returns the raw samples as numpy arrays. Otherwise, they
    are returned as torch tensors.

    Returns:
        a tuple of (transformed_samples, raw_samples) where transformed_samples is a numpy
        array of samples after transforming them to be visualized and raw_samples is either
        a numpy array (if move_raw_to_cpu=True) or a torch tensor (otherwise) that contains the
        HumanML3D vector representation of the samples.

    """
    # create the conditioning dictionary for the prompt string, and given the motion duration
    conditioning_infos = get_prompts_for_model_from_list_of_strings_and_lengths(
        prompts, model_util.batch_size
    )

    # perform sampling given the prepared prompts in batches
    motions = []
    raw_motions = []
    lengths = []
    captions = []
    num_samples = len(prompts)
    num_batches = len(conditioning_infos)

    with torch.no_grad():
        for i, x in enumerate(conditioning_infos):
            print(f"Sampling batch {i + 1}/{num_batches}...")

            # extract the conditioning information that we will use in sampling
            x: T_IDX_COLLATED_DIFFUSION_SAMPLE
            _, model_kwargs, idxs = x

            # the raw sample (T_RAW_SAMPLE_TUPLE)  is a tuple of:
            #
            # (motion_tensor_list, idxs_of_conditioning, num_reps, conditioning_info)
            #
            # mdm output motions are stored in `motion_tensor_list[0]`
            #
            # where, motions.shape: (bs, 263, 1, 196)
            raw_sample: T_RAW_SAMPLE_TUPLE = model_util.generate_raw_samples(
                model_kwargs,
                idxs,
                num_reps=1,
            )

            # we need this output sample in the form: (bs, 22, 3, 196), i.e. as 3D joint positions
            motion_tensor_list, _, _, _ = raw_sample

            # 'raw' motions meaning that these are kept exactly as they are when output from
            # the MDM model, whereas non-raw motions undergo a transformation that may
            # make them more interpretable
            # or useful for some downstream purpose
            s_raw_motions = motion_tensor_list[0]

            # extract some information about the conditioning info from the raw samples
            s_motions, s_lengths, s_captions, _ = model_util.process_raw_samples(
                raw_sample, return_format="to_visualize"
            )

            motions.append(s_motions)
            if move_raw_to_cpu:
                raw_motions.append(s_raw_motions.cpu().numpy())
            else:
                raw_motions.append(s_raw_motions)

            lengths.extend(s_lengths)
            captions.extend(s_captions)

    # extract the motions we care about (non-filler)
    motions = np.concatenate(motions)[:num_samples, :]

    if move_raw_to_cpu:
        raw_motions = np.concatenate(raw_motions)[:num_samples, :]
    else:
        raw_motions = torch.cat(raw_motions)[:num_samples, :]

    return motions, raw_motions


def load_model_from_checkpoint_and_sample(
        prompts: Prompts, model_folder: Path, checkpoint_filename: str = "model.pt"
):
    """Given a list of prompts formatted like: (string description, max_time_in_frames), and
    a model path, sample it and return the results.

    Args:
         prompts:
                these must be a list of tuples of:
                    ("the prompt describing the motion to generate", <int> number of frames
                    for the result)

                for example, the prompt:
                    ("a person walks to their right slowly", 120)

                will generate a motion of 120 frames, conditioned on the given caption.
        model_folder:
        checkpoint_filename:

    Returns:

    """
    # load the MDM model
    model_util = load_mdm_model_from_model_path_for_sampling(
        model_folder,
        checkpoint_filename=checkpoint_filename
    )

    # sample from the model given the supplied prompts
    motions, raw_motions = sample_motion(prompts, model_util)

    # return to the caller
    return motions, raw_motions, model_util


if __name__ == "__main__":
    """This is a simple test to see if model loading and sampling works correctly. 
    """
    # the best model from the MDM repo, for example
    path_name = "exp_outputs/dataset_0064/baseline/generation_50"
    checkpoint_filename = "model000001001.pt"
    my_model_folder = Path(path_name)

    if not my_model_folder.exists():
        raise RuntimeError(
            f"Model folder: {my_model_folder} does not exist. Cannot run basic shape test.")

    # small test to see if the flow works correctly
    my_prompts = [("the person is jumping up and down", 196),
                  ("the person sits down and does not move", 100)]
    motion_result, motion_raw_result, my_model_util = load_model_from_checkpoint_and_sample(
        my_prompts,
        model_folder=my_model_folder,
        checkpoint_filename=checkpoint_filename
    )

    # check shapes and report result
    assert motion_result.shape == (
        2, 22, 3, 196), ("The joint representation was the wrong shape. "
                         "Expected (2, 22, 3, 196)")
    assert motion_raw_result.shape == (
        2, 263, 1, 196), ("The HumanML3D  representation was the wrong shape. "
                          "Expected (2, 263, 1, 196)")
    print("Shapes are correct in small test. Not checking their contents.")

    # larger test to check batching, give a bit of an awkward number to test the padding
    num_prompts_batch_test = 137
    my_prompts = [("the person is jumping up and down", 196)] * num_prompts_batch_test

    # can re-use the already loaded model and checkpoint with this function
    motion_result, motion_raw_result = sample_motion(my_prompts, my_model_util)

    # check shapes
    assert motion_result.shape == (num_prompts_batch_test, 22, 3,
                                   196), (f"The joint representation was the wrong shape. "
                                          f"Expected ({num_prompts_batch_test}, 22, 3, 196)")
    assert motion_raw_result.shape == (num_prompts_batch_test, 263, 1,
                                       196), (f"The HumanML3D  representation was the wrong shape."
                                              f" Expected ({num_prompts_batch_test}, 263, 1, 196)")
    print("Shapes are correct larger test. Not checking their contents.")
