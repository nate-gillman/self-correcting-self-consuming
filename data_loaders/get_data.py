import warnings
from typing import Callable, Optional, Type, Union

from torch.utils.data import DataLoader, Dataset

from data_loaders.humanml.data.dataset import KIT, HumanML3D
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate
from sample.types import (
    T_HUMANML3D_KIT_DATASET_MODE,
    T_HUMANML3D_KIT_DATASET_SPLIT_TYPE,
    T_SUPPORTED_DATASET_NAMES,
)


def get_dataset_class(name: T_SUPPORTED_DATASET_NAMES) -> Type[Dataset]:
    if name == "amass":
        from .amass import AMASS  # type: ignore

        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC

        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses

        return HumanAct12Poses
    elif name == "humanml":
        return HumanML3D
    elif name == "kit":
        return KIT
    else:
        raise ValueError(f"Unsupported dataset name [{name}]")


def get_collate_fn(name: T_SUPPORTED_DATASET_NAMES, hml_mode="train") -> Callable:
    if hml_mode == "gt":
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate

        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        # mode is 'eval'
        # this returns batches as type: T_IDX_COLLATED_DIFFUSION_SAMPLE
        return all_collate


def get_dataset(
    name: T_SUPPORTED_DATASET_NAMES,
    num_frames: int,
    split: T_HUMANML3D_KIT_DATASET_SPLIT_TYPE = "train",
    hml_mode: T_HUMANML3D_KIT_DATASET_MODE = "train",
    subset_by_keyword: Optional[str] = None,
    synthetic_data_dir: Optional[str] = None,
    synthetic_augmentation_percent: Optional[float] = None,
    augmentation_type: Optional[str] = None,
    nearest_neighbor_POC_type: Optional[str] = None,
    is_fully_synthetic: Optional[bool] = False,
    mini_dataset_dir: Optional[str] = None
) -> Union[Dataset, HumanML3D, KIT]:
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(
            split=split,
            num_frames=num_frames,
            mode=hml_mode,
            # if subset_by_keyword = "", set to None
            subset_by_keyword=(subset_by_keyword or None),
            synthetic_data_dir=(synthetic_data_dir or None),
            synthetic_augmentation_percent=(synthetic_augmentation_percent or None),
            augmentation_type=(augmentation_type or None),
            nearest_neighbor_POC_type=(nearest_neighbor_POC_type or None),
            is_fully_synthetic=(is_fully_synthetic or False),
            mini_dataset_dir=(mini_dataset_dir or None)
        )
    else:
        if subset_by_keyword:
            warnings.warn(
                f"Warning: The subset_by_keyword argument was set to: '{subset_by_keyword}', but dataset filtering "
                + f"is implemented only for datasets: humanml, kit. Dataset: {name} is unsupported for filtering."
            )
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(
    name: T_SUPPORTED_DATASET_NAMES,
    batch_size: int,
    num_frames: int,
    split: T_HUMANML3D_KIT_DATASET_SPLIT_TYPE = "train",
    hml_mode: T_HUMANML3D_KIT_DATASET_MODE = "train",
    subset_by_keyword: Optional[str] = None,
    synthetic_data_dir: Optional[str] = None,
    synthetic_augmentation_percent: Optional[float] = None,
    augmentation_type: Optional[str] = None,
    nearest_neighbor_POC_type: Optional[str] = None,
    is_fully_synthetic: Optional[bool] = False,
    mini_dataset_dir: Optional[str] = None,
) -> DataLoader:
    dataset = get_dataset(name, num_frames, split, hml_mode, subset_by_keyword, 
                          synthetic_data_dir, synthetic_augmentation_percent, 
                          augmentation_type, nearest_neighbor_POC_type, is_fully_synthetic,
                          mini_dataset_dir)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        collate_fn=collate,
    )

    return loader
