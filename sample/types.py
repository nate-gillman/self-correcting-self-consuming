"""
The numbers [263|251] are the dimensions of a pose. If using the dataset HumanML3D, there are 263 dimensions.
If using KIT-ML, there are 251 dimensions per pose.

A human motion sequence is an array of poses. So we may expect the shape of a pose sequence to be:
    (seq_length, [263|251]).

Add the batch size, and we expect our data to resemble an object like:
    (batch_size, seq_length, [263|251])


"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from typing_extensions import Literal, TypedDict

T_BATCH = Any
T_RAW_DIFFUSION_SAMPLE = torch.Tensor
T_IN_MEMORY_TRAJ_SAMPLE = Tuple[str, List[str], np.ndarray, np.ndarray]
T_VISUALIZABLE_TRAJ_SAMPLE = torch.Tensor
T_SUPPORTED_DATASET_NAMES = Literal["amass", "uestc", "humanact12", "humanml", "kit"]


class MotionTextSampleDict(TypedDict):
    # human-annotated semantic descriptions of a motion
    # ex. 'a person who has their arms up and then puts there arms down and steps back and lunges forward'
    caption: str
    # part of speech (POS) tags for word vectorizor
    # ex. [
    #   'a/DET', 'person/NOUN', 'who/PRON', 'has/AUX', 'their/DET',
    #   'arm/NOUN', 'up/ADV', 'and/CCONJ', ..., 'forward/ADV'
    # ]
    tokens: List[str]


class MotionSequenceSampleDict(TypedDict):
    """
    Objects of this form are how HumanML3D data points are stored on disk and in Text2MotionDatasetV2.data_dict

    This is not the form that samples will take when given by Text2MotionDatasetV2.__getitem__ because
    they are first given to a collate function (data_loaders.tensors.[t2m_collate|collate|collate_fn]).

    The collate function transforms the data representation on disk to what the model 'sees' when
    it is training. See example in: train/training_loop.py run_loop().
    """
    motion: np.ndarray  # (seq_length, [263|251])
    length: int  # seq_length

    # a list of caption objects and their tokenization
    text: List[MotionTextSampleDict]  # len = number of captions, e.g. 3-5


# how data are represented behind the scenes in memory within
# Text2MotionDatasetV2
# maps datapoint name (str -> datapoint)
# names are strings that look like: 'M013217', 'M013218', etc. or '6d66301d29cd996b3b26c713cb127148_M013854'
# where the last example is the md5 hash of md5(start_frame, end_frame, caption, parent_sequence_filename)
T_DATA_DICT = Dict[str, MotionSequenceSampleDict]


class DiffusionConditioningDict(TypedDict):
    """
    This is information used for conditional diffusion. Most of the work
    to create these objects is performed in data_loaders/tensors.py -> collate

    a DataLoader's iterator will return b objects, which will be collated into tensors of
    shape: (b, ., ., .). During collation, the below fields are derived from Text2MotionDatasetV2.data_dict,
    which is of type: T_DATA_DICT
    """

    mask: torch.Tensor

    # these are all given in the dataset, they are just lightly transformed to
    # become the below:
    lengths: torch.Tensor
    text: List[str]
    tokens: List[List[str]]

    action: Optional[torch.Tensor]
    action_text: Optional[List[str]]


# example of T_IDX_COLLATED_DIFFUSION_SAMPLE, returned by a DataLoader iterator. This is
# how samples stored on disk look after they are transformed by __getitem__ and collate functions.
# The model will 'see' something like the below during training:
#
# tensor: [batch_size, [263|251], 1, seq_length]
# dict: {'y': {
#     'mask': tensor: [batch_size, 1, 1, seq_length]
#     'lengths': tensor [batch_size]
#     'text': list of string (len = batch_size)
#     'tokens': list of string (len = batch_size)
#
#      ?'action': [tensor]
#      ?'action_text': [str]
# }}
# tensor: [batch_size, 1], the indices of the samples
T_MODEL_KWARGS = Dict[str, DiffusionConditioningDict]
T_IDX_COLLATED_DIFFUSION_SAMPLE = Tuple[torch.Tensor, T_MODEL_KWARGS, torch.Tensor]


class GeneratedSampleBatchDict(TypedDict):
    motion: np.ndarray
    text: List[List[str]]
    lengths: np.ndarray
    num_samples: int
    num_repetitions: int
    # for autophagous model, need to know the indices of the samples corresponding
    # to the conditioning information we use to generate new ones
    conditioning_idxs: torch.Tensor


T_RAW_SAMPLE_TUPLE = Tuple[
    List[T_RAW_DIFFUSION_SAMPLE], torch.Tensor, int, T_MODEL_KWARGS
]

T_HUMANML3D_KIT_DATASET_SPLIT_TYPE = Literal["test", "train", "val", "train_val"]
T_SAMPLE_RETURN_FORMAT = Literal[
    "in_memory", "to_visualize", "save_raw_motion", "save_generated_motion"
]
T_HUMANML3D_KIT_DATASET_MODE = Literal[
    "gt",
    "train",
    "eval",
    "text_only",
]

#  Prompts:
#         these must be a list of tuples of:
#             ("the prompt describing the motion to generate", <int> number of frames for the result)
#
#         for example, the prompt:
#             ("a person walks to their right slowly", 120)
#
#         will generate a motion of 120 frames, conditioned on the given caption.
Prompts = List[Tuple[str, int]]
