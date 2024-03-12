import os
from abc import ABC
from argparse import Namespace
from numbers import Number
from typing import Optional


class TrainPlatform(ABC):
    def __init__(self, save_dir: str) -> None:
        pass

    def report_scalar(
        self, name: str, value: Number, iteration: int, group_name: Optional[str] = None
    ) -> None:
        pass

    def report_args(self, args: Namespace, name: str) -> None:
        pass

    def close(self) -> None:
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir: str) -> None:
        from clearml import Task  # type: ignore

        path, name = os.path.split(save_dir)
        self.task = Task.init(
            project_name="motion_diffusion", task_name=name, output_uri=path
        )
        self.logger = self.task.get_logger()

    def report_scalar(
        self, name: str, value: Number, iteration: int, group_name: str
    ) -> None:
        self.logger.report_scalar(
            title=group_name, series=name, iteration=iteration, value=value
        )

    def report_args(self, args: Namespace, name: str) -> None:
        self.task.connect(args, name=name)

    def close(self) -> None:
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(
        self, name: str, value: Number, iteration: int, group_name: Optional[str] = None
    ) -> None:
        self.writer.add_scalar(f"{group_name}/{name}", value, iteration)

    def close(self) -> None:
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir: Optional[str] = None) -> None:
        self.save_dir = save_dir


def get_train_platform(args: Namespace) -> TrainPlatform:
    # configure the training platform
    platform_type = args.train_platform_type
    if platform_type == "ClearmlPlatform":
        train_platform = ClearmlPlatform(args.save_dir)
    elif platform_type == "TensorboardPlatform":
        train_platform = TensorboardPlatform(args.save_dir)
    else:
        train_platform = NoPlatform(args.save_dir)
    train_platform.report_args(args, name="Args")
    return train_platform
