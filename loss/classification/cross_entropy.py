from argparse import ArgumentParser, Namespace
from typing import Dict

from torch import Tensor
from torch.nn import functional as F

from loss import LOSS_REGISTRY
from loss.classification.base_classification_loss import BaseClassificationLoss


@LOSS_REGISTRY.register(name="cross_entropy", class_type="classification")
class CrossEntropyLoss(BaseClassificationLoss):
    def __init__(self, cfg: Namespace, *args, **kwargs) -> None:
        super().__init__(cfg=cfg, *args, **kwargs)

        self.ignore_index = getattr(
            cfg, "loss.classification.cross_entropy.ignore_index"
        )
        self.label_smoothing = getattr(
            cfg, "loss.classification.cross_entropy.label_smoothing"
        )

    def compute_loss(self, prediction: Tensor, target: Tensor):
        """
        Copmutes the loss given the model's outputs and target labels.
        """
        if isinstance(prediction, Dict):
            logits = prediction["logits"]
        else:
            logits = prediction
        return F.cross_entropy(
            input=logits,
            target=target,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        if cls != CrossEntropyLoss:
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--loss.classification.cross-entropy.ignore-index",
            type=int,
            default=-1,
            help="The class/label index to ignore. Defaults to -1.",
        )

        group.add_argument(
            "--loss.classification.cross-entropy.label-smoothing",
            type=float,
            default=0.0,
            help="A vlaue between 0 and 1 indicating how much label smoothing to apply. Defaults to 0.",
        )

        return parser

    def __repr__(self) -> str:
        repr_str = (
            f"{self.__class__.__name__}("
            f"\n\t ignore_index={self.ignore_index},"
            f"\n\t label_smoothing={self.label_smoothing})"
        )

        return repr_str
