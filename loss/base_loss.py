from argparse import ArgumentParser, Namespace
from typing import Dict, Optional, Union

from torch import Tensor, nn


class BaseLoss(nn.Module):
    def __init__(self, cfg: Namespace, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.cfg = cfg

    def forward(
        self,
        prediction: Union[Dict, Tensor],
        target: Tensor,
        inputs: Optional[Union[Dict, Tensor]] = None,
    ) -> Tensor:
        """
        Computes the loss using the given predictions and targets. Optionally
        makes use of the input samples used to obtain the predictions.

        Args:
            prediction: The output of the model. Can be a tensor or dictionary.
            target: The target label.
            inputs: The input to the model. Can be a a tensor or a dictionary (e.g.,
            if doing multi-modal training). If None, assumes it won't be used.
        """

        raise NotImplementedError()

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        if cls != BaseLoss:
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--loss.category",
            type=str,
            default=None,
            help="The category of the loss function to use. Defaults to None.",
        )

        return parser
