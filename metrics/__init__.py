from argparse import ArgumentParser, Namespace

from metrics.base_metric import BaseMetric
from utils.registry import Registry

METRIC_REGISTRY = Registry(
    name="metrics",
    subdirs=["metrics"],
)


def get_metric_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser = BaseMetric.add_arguments(parser)
    parser = METRIC_REGISTRY.get_all_arguments(parser)

    return parser


def get_metric_from_registry(
    cfg: Namespace, name: str, mode: str, *args, **kwargs
) -> BaseMetric:
    metric = METRIC_REGISTRY[f"metric.{name}"](cfg, mode=mode, *args, **kwargs)

    return metric
