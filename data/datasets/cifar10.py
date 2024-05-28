from argparse import Namespace

from data.datasets import DATASET_REGISTRY
from data.datasets.base_cifar import CifarDataset


@DATASET_REGISTRY.register(name="cifar10", class_type="classification")
class Cifar10Dataset(CifarDataset):
    def __init__(
        self,
        cfg: Namespace,
        mode: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(cfg=cfg, mode=mode, label_key=b"labels", *args, **kwargs)
