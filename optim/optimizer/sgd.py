from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable, Union

from torch import Tensor
from torch.optim import SGD

from logger import logger
from optim.optimizer import OPTIM_REGISTRY
from optim.optimizer.base_optimizer import BaseOptimizer
from utils.loading import load_from_torch


@OPTIM_REGISTRY.register(name="sgd", class_type="optimizer")
class SGDOptimizer(BaseOptimizer, SGD):
    def __init__(
        self,
        cfg: Namespace,
        model_params: Iterable[Union[Tensor, Dict]],
        *args,
        **kwargs,
    ) -> None:
        BaseOptimizer.__init__(self, cfg)

        momentum = getattr(cfg, "optim.sgd.momentum")
        nesterov = getattr(cfg, "optim.sgd.nesterov")

        SGD.__init__(
            self,
            params=model_params,
            lr=0,  # This will get set by the scheduler
            momentum=momentum,
            weight_decay=self.weight_decay,
            nesterov=nesterov,
        )

        self.load_from_state_dict()

    def load_from_state_dict(self):
        if self.state_dict_path is not None:
            state_dict = load_from_torch(self.state_dict_path)

            state_dict = state_dict[self.state_dict_key]
            logger.info(f"Loaded optimizer state dict from {self.state_dict_path}")

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--optim.sgd.momentum",
            type=float,
            default=0.9,
            help="The SGD momentum factor.",
        )

        group.add_argument(
            "--optim.sgd.nesterov",
            action="store_true",
            help="Whether to enable Nesterov momentum or not. Defaults to False.",
        )

        return parser
