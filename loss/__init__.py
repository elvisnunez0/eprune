from argparse import ArgumentParser, Namespace

from loss.base_loss import BaseLoss
from utils.registry import Registry

LOSS_REGISTRY = Registry(
    name="loss",
    subdirs=["loss"],
)


def get_loss_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser = BaseLoss.add_arguments(parser)
    parser = LOSS_REGISTRY.get_all_arguments(parser)

    return parser


def get_loss_from_registry(cfg: Namespace, *args, **kwargs):
    loss_category = getattr(cfg, "loss.category")
    loss_name = getattr(cfg, f"loss.{loss_category}.name")
    model = LOSS_REGISTRY[f"{loss_category}.{loss_name}"](cfg, *args, **kwargs)

    return model
