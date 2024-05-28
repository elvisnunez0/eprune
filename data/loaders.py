from argparse import Namespace

from torch.utils.data import DataLoader

from data.datasets import get_dataset_from_registry


def create_loader(cfg: Namespace, mode: str, shuffle: bool = True) -> DataLoader:
    dataset = get_dataset_from_registry(cfg=cfg, mode=mode)

    if "train" in mode:
        batch_size = getattr(cfg, "dataset.batch_size_train")
    else:
        batch_size = getattr(cfg, "dataset.batch_size_test")
    num_workers = getattr(cfg, "dataset.num_workers")
    pin_memory = getattr(cfg, "dataset.pin_memory")

    # NOTE: Will use standard sampler.
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader
