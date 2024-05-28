from argparse import ArgumentParser, Namespace
from typing import Dict, Union

from loguru._logger import Logger
from torch import Tensor


class BaseMetric(object):
    def __init__(self, cfg: Namespace) -> None:
        self.cfg = cfg

    def reset(self) -> None:
        raise NotImplementedError()

    def save(self, logger: Logger) -> None:
        raise NotImplementedError()

    def update(self, prediction: Union[Dict, Tensor], target: Tensor) -> None:
        raise NotImplementedError()

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        if cls != BaseMetric:
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--metrics.train",
            type=tuple,
            default=["loss"],
            help="The metrics to track for the train set.",
        )

        group.add_argument(
            "--metrics.test",
            type=tuple,
            default=["loss"],
            help="The metrics to track for the test set.",
        )

        return parser
