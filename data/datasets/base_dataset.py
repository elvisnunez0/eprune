import argparse
from typing import Tuple, Union

from torch.utils.data import Dataset

from data.transforms.base_transform import BaseTransform


class BaseDataset(Dataset):
    """
    This is a base class that should be extended by all other dataset classes.
    It contains functions that should be implemented by all other dataset classes.

    Args:
        cfg: Config arguments.
        mode: Indicated whether the model is in train, test, or validation mode.
    """

    def __init__(
        self,
        cfg: argparse.Namespace,
        mode: str,
    ) -> None:
        super().__init__()

        assert mode in [
            "train",
            "train_no_aug",
            "test",
            "validation",
        ], "mode must be one of train, train_no_aug, test, or validation."

        self.cfg = cfg
        self.mode = mode
        self.root_expanduser = getattr(cfg, "dataset.root_expanduser")

    def train_transforms(self, size: Union[int, Tuple[int, int]] = None):
        """
        The transformations to apply to training samples.
        """
        raise NotImplementedError()

    def test_transforms(self):
        """
        The transformations to apply to test samples.
        """
        raise NotImplementedError()

    def validation_transforms(self):
        """
        The transformations to apply to validation samples.
        """
        raise NotImplementedError()

    def _get_transforms(
        self, size: Union[int, Tuple[int, int]] = None
    ) -> BaseTransform:
        if self.mode == "train":
            return self.train_transforms(size=size)
        elif self.mode == "test" or self.mode == "train_no_aug":
            return self.test_transforms()
        elif self.mode == "validation":
            return self.validation_transforms()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BaseDataset:
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--dataset.name",
            type=str,
            default="",
            help="The name of the dataset being used.",
        )

        group.add_argument(
            "--dataset.root-expanduser",
            action="store_true",
            help="If true, will append the user's home directory to the "
            "root train/test directories. Defaults to False.",
        )

        group.add_argument(
            "--dataset.root-train",
            type=str,
            help="The directory containing training samples. Defaults to None.",
        )

        group.add_argument(
            "--dataset.root-test",
            type=str,
            help="The directory containing testing samples.",
        )

        group.add_argument(
            "--dataset.root-validation",
            type=str,
            help="The directory containing validation samples.",
        )

        group.add_argument(
            "--dataset.train-batch-size",
            type=int,
            default=128,
            help="The batch size used for training. If distributed training, this is the batch size per GPU.",
        )

        group.add_argument(
            "--dataset.test-batch-size",
            type=int,
            default=100,
            help="The batch size used for testing.",
        )

        group.add_argument(
            "--dataset.validation-batch-size",
            type=int,
            default=100,
            help="The batch size used for validation.",
        )

        group.add_argument(
            "--dataset.num-workers",
            type=int,
            default=1,
            help="The number of data workers. Defaults to 1.",
        )

        group.add_argument(
            "--dataset.persistant-workers",
            action="store_true",
            help="Whether to use the same workers when loading the data. Defaults to False.",
        )

        group.add_argument(
            "--dataset.pin-memory",
            action="store_true",
            help="Whether data should be pinned in memory for data loaders. Defaults to False.",
        )

        return parser
