from argparse import Namespace

from data.datasets import DATASET_REGISTRY
from data.datasets.base_cifar import CifarDataset


@DATASET_REGISTRY.register(name="cifar20", class_type="classification")
class Cifar20Dataset(CifarDataset):
    """
    This class uses the superclass/'coarse' labels of the CIFAR100 dataset.
    """

    def __init__(
        self,
        cfg: Namespace,
        mode: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            cfg=cfg, mode=mode, label_key=b"coarse_labels", *args, **kwargs
        )
