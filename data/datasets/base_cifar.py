import os
import pickle
from argparse import Namespace
from typing import Any, Dict, Tuple, Union

import numpy as np
from PIL import Image

from data.datasets.base_dataset import BaseDataset
from data.transforms.image_transforms import (Normalize, RandomCrop,
                                              RandomHorizontalFlip, Resize,
                                              ToTensor)
from data.transforms.universal_transforms import Compose
from saving.utils import create_dir


class CifarDataset(BaseDataset):
    """
    This class serves as a base class for the CIFAR10 and CIFAR100
    datasets.
    """

    def __init__(
        self,
        cfg: Namespace,
        mode: str,
        label_key: str,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            cfg: Config arguments.
            mode: Either 'train' or 'test' specifying whether to load the
                train or test sets.
            label_key: When reading the pickled CIFAR train/test files, we
                obtain a dict containnig the data labels. This argument
                refers to the name of the key that contains the labels,
                which differs for CIFAR10 and CIFAR100.
        """
        super().__init__(cfg=cfg, mode=mode, *args, **kwargs)

        if "train" in self.mode:
            self.root = getattr(cfg, "dataset.root_train")
        elif "test" in self.mode:
            self.root = getattr(cfg, "dataset.root_test")
        else:
            raise ValueError()

        if self.root is None:
            name = getattr(cfg, "dataset.name")
            self.root = self._download_dataset(name)

        if self.root_expanduser:
            self.root = os.path.expanduser(f"~/{self.root}")

        self.label_key = label_key
        data, labels = self._load_batches()

        self.data = data
        self.labels = labels

    def _download_dataset(self, name: str):
        # TODO
        pass

    def _unpickle(self, file: str) -> dict:
        """
        Unpickles the file at the provided path.

        Args:
            file: The path to the file to unpickle.

        Returns:
            dict: The unpickled file.
        """
        with open(file, "rb") as fo:
            unpickleed_file = pickle.load(fo, encoding="bytes")
        return unpickleed_file

    def _load_batches(self) -> np.array:
        """
        Reads the pickled batch files in @self.root and concatenates them into
        a single numpy array.

        For CIFAR 10, assumes the train folder contains five files called
            'data_batch_k' where k is 1, 2, ..., 5. Assumes the test folder
            contains 'test_batch.'

        For CIFAR100, assumes the train folder contains a single file called
            'train' and the test folder contains a single file called 'test.'

        Returns:
            data: If @self.root is the training directory, returns a NumPy array with shape
                [50000, 3, 32, 32]. If @self.root is the test directory, returns a NumPy array
                with shape [10000, 3, 32, 32].
            labels: A list of length 50000 if @self.root is training. If @self.root is
             test, a list of length 10000. Each element is in the set {0, 1, ..., 9} for
             CIFAR10 and {0, 1, ..., 99} for CIFAR100.
        """
        batch_files = os.listdir(self.root)
        data = None
        labels = []
        for file in batch_files:
            batch_file = os.path.join(self.root, file)
            batch_dict = self._unpickle(batch_file)
            batch_arr = batch_dict[b"data"]

            B, _ = batch_arr.shape
            batch_arr = batch_arr.reshape((B, 3, 32, 32))
            batch_labels = batch_dict[self.label_key]

            if data is None:
                data = batch_arr
                labels = batch_labels
            else:
                data = np.vstack([data, batch_arr])
                labels = labels + batch_labels

        return data, labels

    def train_transforms(
        self, size: Union[int, Tuple[int, int]] = None, *args, **kwargs
    ):
        """
        Returns transformations applied to the input in training mode.

        Args:
            size: The resize resolution. If None, will read the resize from the command line arguments.
        """
        transforms_list = []

        if getattr(self.cfg, "dataset.image_transforms.random_crop.enable"):
            transforms_list.append(RandomCrop(cfg=self.cfg))

        if getattr(self.cfg, "dataset.image_transforms.random_horizontal_flip.enable"):
            transforms_list.append(RandomHorizontalFlip(cfg=self.cfg))

        if getattr(self.cfg, "dataset.image_transforms.resize.enable"):
            transforms_list.append(Resize(cfg=self.cfg))

        transforms_list.append(ToTensor(cfg=self.cfg))

        if getattr(self.cfg, "dataset.image_transforms.normalize.enable"):
            transforms_list.append(Normalize(cfg=self.cfg))

        return Compose(cfg=self.cfg, transforms_list=transforms_list)

    def test_transforms(self, *args, **kwargs):
        """
        Returns transformations applied to the input during testing mode.
        """
        transforms_list = []

        if getattr(self.cfg, "dataset.image_transforms.resize.enable"):
            transforms_list.append(Resize(cfg=self.cfg))

        transforms_list.append(ToTensor(cfg=self.cfg))

        if getattr(self.cfg, "dataset.image_transforms.normalize.enable"):
            transforms_list.append(Normalize(cfg=self.cfg))

        return Compose(cfg=self.cfg, transforms_list=transforms_list)

    def __getitem__(
        self, sample_inf: Union[Tuple[int, int, int], int]
    ) -> Dict[str, Any]:
        """
        Returns the sample corresponding to the input sample index.

        Args:
            sample_inf: Either a tuple or an int. If a tuple, contains
                (crop_size_h, crop_size_w, index) which is useful if we
                want to train at different resolutions. If an int, only
                contains the index of the sample to return.

        Returns:
            data: A dictionary containing:
                sample: A tensor containing the sample image with shape [C, H, W].
                target: An integer specifying the sample's label.
                sample_id: The sample index.
        """

        if isinstance(sample_inf, tuple):
            index = sample_inf[2]
            size = sample_inf[:2]
        else:
            index = sample_inf
            size = None

        transform_fn = self._get_transforms(size=size)

        # Shape is [C, H, W] and each element is in [0, 255] range.
        input_img = self.data[index]

        # Transform numpy array to PIL image
        input_img = Image.fromarray(input_img.astype("uint8").transpose(1, 2, 0), "RGB")
        target = self.labels[index]

        data = {"image": input_img}
        data = transform_fn(data, key="image")

        data["sample"] = data.pop("image")
        data["target"] = target
        data["sample_id"] = index

        return data

    def __len__(self) -> int:
        return len(self.labels)
