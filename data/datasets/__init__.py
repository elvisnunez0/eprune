from argparse import ArgumentParser, Namespace

from data.datasets.base_dataset import BaseDataset
from utils.registry import Registry

DATASET_REGISTRY = Registry(
    name="datasets",
    subdirs=["data/datasets"],
)


def get_dataset_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser = BaseDataset.add_arguments(parser)
    parser = DATASET_REGISTRY.get_all_arguments(parser)

    return parser


def get_dataset_from_registry(
    cfg: Namespace, mode: str, *args, **kwargs
) -> BaseDataset:
    dataset_name = getattr(cfg, "dataset.name")
    task = getattr(cfg, "common.task")
    dataset = DATASET_REGISTRY[f"{task}.{dataset_name}"](cfg, mode, *args, **kwargs)

    return dataset
