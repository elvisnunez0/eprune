from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable, Union

from torch import Tensor

from utils.registry import Registry

OPTIM_REGISTRY = Registry(
    name="optimizers",
    subdirs=["optim/optimizer"],
)


def get_optim_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser = OPTIM_REGISTRY.get_all_arguments(parser)

    return parser


def get_optim_from_registry(
    cfg: Namespace, model_params: Iterable[Union[Tensor, Dict]], *args, **kwargs
):
    optim_name = getattr(cfg, "optim.name")
    model = OPTIM_REGISTRY[f"optimizer.{optim_name}"](
        cfg, model_params=model_params, *args, **kwargs
    )

    return model
