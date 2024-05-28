from argparse import ArgumentParser, Namespace
from typing import Dict, Optional, Union

from torch import Tensor

from loss import LOSS_REGISTRY
from loss.base_loss import BaseLoss


@LOSS_REGISTRY.register(name="__base__", class_type="classification")
class BaseClassificationLoss(BaseLoss):
    def __init__(self, cfg: Namespace, *args, **kwargs) -> None:
        super().__init__(cfg=cfg, *args, **kwargs)

    def compute_loss(self, prediction: Tensor, target: Tensor):
        """
        Copmutes the loss given the model's outputs and target labels.
        """
        raise NotImplementedError

    def forward(
        self,
        prediction: Union[Dict, Tensor],
        target: Tensor,
        inputs: Optional[Union[Dict, Tensor]] = None,
    ) -> Tensor:
        if isinstance(prediction, Dict):
            assert (
                "logits" in prediction
            ), "Logits must be provided for classification loss."
            return self.compute_loss(prediction=prediction["logits"], target=target)
        elif isinstance(prediction, Tensor):
            return self.compute_loss(prediction=prediction, target=target)

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        if cls != BaseClassificationLoss:
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--loss.classification.name",
            type=str,
            default=None,
            help="The name of the classification loss. Defaults to None.",
        )

        return parser
