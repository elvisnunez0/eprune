import os
from argparse import Namespace
from typing import Any, Dict, List, Tuple, Union

from PIL import Image

from data.datasets import DATASET_REGISTRY
from data.datasets.base_dataset import BaseDataset
from data.transforms.image_transforms import (Normalize, RandomCrop,
                                              RandomHorizontalFlip, Resize,
                                              ToTensor)
from data.transforms.universal_transforms import Compose


@DATASET_REGISTRY.register(name="tiny_imagenet", class_type="classification")
class TinyImageNet(BaseDataset):
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
        elif "test" in self.mode:
            self.root = getattr(cfg, f"dataset.root_test")
        else:
            raise ValueError()

        if self.root_expanduser:
            self.root = os.path.expanduser(f"~/{self.root}")

        self.img_path_tuples = self.construct_img_label_tuples()

    def construct_img_label_tuples(self) -> List[Tuple[str, int]]:
        """
        Constructs a list of tuples of the form
            (path to image, integer label)
        """
        folders = os.listdir(self.root)
        folders = sorted([f for f in folders if f[0] != "."])
        int_label_map = {folder: label for label, folder in enumerate(folders)}

        path_label_tuples = []
        for folder in folders:
            if folder[0] != ".":
                img_dir = os.path.join(self.root, folder, "images")
                imgs = os.listdir(img_dir)
                int_label = int_label_map[folder]
                for img_name in imgs:
                    img_path = os.path.join(img_dir, img_name)
                    path_label_tuples.append((img_path, int_label))

        return path_label_tuples

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
