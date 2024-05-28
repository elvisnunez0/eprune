from argparse import Namespace
from typing import Dict, Union

from loguru._logger import Logger
from torch import Tensor

from metrics import get_metric_from_registry

METRIC_TO_CLASS_MAP = {
    "top1": "accuracy",
    "top5": "accuracy",
    "sphere_projection": "sphere_projection",
    "cross_entropy": "cross_entropy",
}
METRIC_PRINT_MAP = {"top1": "Top-1", "top5": "Top-5", "cross_entropy": "Cross Entropy"}


class MetricComposer(object):
    def __init__(self, cfg: Namespace, mode: str) -> None:
        self.metric_names = sorted(getattr(cfg, f"metrics.{mode}"))
        self.cfg = cfg
        self.mode = mode

        self._construct_metric_modules_list()

    def _construct_metric_modules_list(self):
        self.metric_modules = []
        for name in self.metric_names:
            class_name = METRIC_TO_CLASS_MAP[name]
            self.metric_modules.append(
                get_metric_from_registry(cfg=self.cfg, name=class_name, mode=self.mode)
            )

    def update(self, prediction: Union[Dict, Tensor], target: Tensor) -> None:
        for metric_module in self.metric_modules:
            metric_module.update(prediction=prediction, target=target)

    def get_metric_dict(self):
        merged_metric_dict = {}
        for metric_module in self.metric_modules:
            merged_metric_dict.update(metric_module.get_metrics())

        return merged_metric_dict

    def reset(self) -> None:
        for metric_module in self.metric_modules:
            metric_module.reset()

    def save(self, logger: Logger) -> None:
        for metric_module in self.metric_modules:
            metric_module.save(logger=logger)

    def log_metrics(self, logger: Logger, prefix: str = ""):
        merged_metric_dict = self.get_metric_dict()

        # Form single string with all metrics
        cat_metrics = prefix
        for metric_name, metric in merged_metric_dict.items():
            print_name = METRIC_PRINT_MAP[metric_name]
            cat_metrics = f"{cat_metrics} | {print_name}: {metric:0.4f}"

        logger.info(cat_metrics)
