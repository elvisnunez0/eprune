from argparse import ArgumentParser, Namespace

from optim.lr_scheduler.base_lr_scheduler import BaseLRScheduler
from utils.registry import Registry

LR_SCHEDULER_REGISTRY = Registry(
    name="lr_scheduler",
    subdirs=["optim/lr_scheduler"],
)


def get_lr_scheduler_arguments(parser: ArgumentParser) -> ArgumentParser:
    # Add base LR args
    parser = BaseLRScheduler.add_arguments(parser)
    parser = LR_SCHEDULER_REGISTRY.get_all_arguments(parser)

    return parser


def get_lr_scheduler_from_registry(cfg: Namespace, *args, **kwargs) -> BaseLRScheduler:
    scheduler_name = getattr(cfg, "optim.lr_scheduler.name")
    lr_scheduler = LR_SCHEDULER_REGISTRY[f"lr_scheduler.{scheduler_name}"](
        cfg, *args, **kwargs
    )

    return lr_scheduler
