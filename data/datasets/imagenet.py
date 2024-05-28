import argparse
import os
from argparse import Namespace
from typing import Any, Dict, Tuple, Union

import numpy as np
from PIL import Image

from data.datasets import DATASET_REGISTRY
from data.datasets.base_dataset import BaseDataset
from data.transforms.image_transforms import (CenterCrop, Normalize,
                                              RandomHorizontalFlip,
                                              RandomResizedCrop, Resize,
                                              ToTensor)
from data.transforms.universal_transforms import Compose


@DATASET_REGISTRY.register(name="imagenet", class_type="classification")
class ImageNetDataset(BaseDataset):
    """
    This class defines the ImageNet dataset.
    """

    def __init__(
        self,
        cfg: Namespace,
        mode: str,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            cfg: Config arguments.
            mode: Either 'train' or 'test' specifying whether to load the
                train or test sets.
        """
        super().__init__(cfg=cfg, mode=mode, *args, **kwargs)

        if "train" in self.mode:
            self.root = getattr(cfg, f"dataset.root_train")
        elif (
            "test" in self.mode
        ):  # The validation set is typically treated as the test set in ImageNet
            self.root = getattr(cfg, f"dataset.root_test")
        else:
            raise ValueError()

        if self.root_expanduser:
            self.root = os.path.expanduser(f"~/{self.root}")

        self.synset_mapping = self.read_synset_mapping(cfg)

        if "train" in self.mode:
            self.img_path_tuples = self.get_train_mapping(cfg)
        else:
            self.img_path_tuples = self.get_val_mapping(cfg)

    def read_synset_mapping(self, cfg: Namespace) -> dict:
        path = getattr(cfg, "dataset.imagenet.synset_mapping")

        # Folder name to index label
        mapping = {}

        # Each row is assumed to have the form 'n01440764 <label text label, e.g., tench, Tinca tinca>'
        with open(path, "r") as f:
            for i, line in enumerate(f.readlines()):
                folder = line.split(" ")[0]
                mapping[folder] = i

        return mapping

    def read_val_labels(self, cfg: Namespace) -> dict:
        path = getattr(cfg, "dataset.imagenet.val_labels")
        synset_mapping = self.synset_mapping

        # Map validation file name to
        file_name_to_label = {}

        with open(path, "r") as f:
            for line in f.readlines():
                if "ILSVRC2012_val" in line:
                    file_name, rem_str = line.split(",")
                    folder = rem_str.split(" ")[0]
                    label = synset_mapping[folder]
                    file_name_to_label[file_name] = label

        return file_name_to_label

    def get_train_mapping(self, cfg: Namespace) -> dict:
        synset_mapping = self.read_synset_mapping(cfg)

        img_extns = [".jpeg", ".jpg", ".png"]
        mapping = {}
        for root, _, files in os.walk(self.root):
            for file in files:
                filename, extension = os.path.splitext(file)
                if extension.lower() in img_extns:
                    folder = filename.split("_")[0]
                    int_label = synset_mapping[folder]
                    img_path = os.path.join(root, file)
                    mapping[img_path] = int_label

        flattened_map = [(img_path, label) for img_path, label in mapping.items()]

        return flattened_map

    def get_val_mapping(self, cfg: Namespace) -> dict:
        val_labels_mapping = self.read_val_labels(cfg)

        img_extns = [".jpeg", ".jpg", ".png"]
        mapping = {}
        for root, _, files in os.walk(self.root):
            for file in files:
                filename, extension = os.path.splitext(file)
                if extension.lower() in img_extns:
                    int_label = val_labels_mapping[filename]
                    img_path = os.path.join(root, file)
                    mapping[img_path] = int_label

        flattened_map = [(img_path, label) for img_path, label in mapping.items()]

        return flattened_map

    def train_transforms(
        self, size: Union[int, Tuple[int, int]] = None, *args, **kwargs
    ):
        """
        Returns transformations applied to the input in training mode.

        Args:
            size: The resize resolution. If None, will read the resize from the command line arguments.
        """
        transforms_list = []

        if getattr(self.cfg, "dataset.image_transforms.random_resized_crop.enable"):
            transforms_list.append(RandomResizedCrop(cfg=self.cfg))

        if getattr(self.cfg, "dataset.image_transforms.random_horizontal_flip.enable"):
            transforms_list.append(RandomHorizontalFlip(cfg=self.cfg))

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

        if getattr(self.cfg, "dataset.image_transforms.center_crop.enable"):
            transforms_list.append(CenterCrop(cfg=self.cfg))

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

        img_path, target = self.img_path_tuples[index]
        input_img = Image.open(img_path).convert("RGB")

        data = {"image": input_img}
        data = transform_fn(data, key="image")

        data["sample"] = data.pop("image")
        data["target"] = target
        data["sample_id"] = index

        return data

    def __len__(self) -> int:
        return len(self.img_path_tuples)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != ImageNetDataset:
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--dataset.imagenet.synset-mapping",
            type=str,
            default="",
            help="The directory to the LOC_synset_mapping.txt file which maps folder names to labels.",
        )

        group.add_argument(
            "--dataset.imagenet.val-labels",
            type=str,
            default="",
            help="The directory to the LOC_val_solution.csv file which maps validation file names to folder names.",
        )

        return parser
