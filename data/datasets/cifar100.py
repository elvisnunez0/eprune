from argparse import Namespace

from data.datasets import DATASET_REGISTRY
from data.datasets.base_cifar import CifarDataset


@DATASET_REGISTRY.register(name="cifar100", class_type="classification")
class Cifar100Dataset(CifarDataset):
    def __init__(
        self,
        cfg: Namespace,
        mode: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(cfg=cfg, mode=mode, label_key=b"fine_labels", *args, **kwargs)
