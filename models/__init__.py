from argparse import ArgumentParser, Namespace

from models.base_model import BaseModel
from models.pruners.ustruct_pruner import UStructPruner
from utils.registry import Registry

MODEL_REGISTRY = Registry(
    name="models",
    subdirs=["models"],
)


def get_model_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser = BaseModel.add_arguments(parser)
    parser = UStructPruner.add_arguments(parser)
    parser = MODEL_REGISTRY.get_all_arguments(parser)
    return parser


def get_model_from_registry(cfg: Namespace, *args, **kwargs) -> BaseModel:
    task = getattr(cfg, "common.task")
    model_name = getattr(cfg, f"model.{task}.name")
    model = MODEL_REGISTRY[f"{task}.{model_name}"](cfg, *args, **kwargs)

    return model
