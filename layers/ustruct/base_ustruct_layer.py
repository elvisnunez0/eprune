from argparse import ArgumentParser
from typing import Dict

from torch import Tensor


class BasePruneModule(object):
    """
    Base class for prunable modules with weight and bias buffers.
    """

    def __init__(
        self,
        prune_bias: bool = False,
        sparsity_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.prune_bias = prune_bias
        self.sparsity_scale = sparsity_scale

    def is_prunable(self) -> bool:
        return True

    def register_mask_buffers(self) -> None:
        self.register_buffer("mask_w", None)
        self.register_buffer("mask_b", None)

    def pruned_weight(self) -> Tensor:
        if self.mask_w is not None:
            return self.weight * self.mask_w

        return self.weight

    def pruned_bias(self) -> Tensor:
        if self.mask_b is not None:
            return self.bias * self.mask_b

        return self.bias

    def count_zeros(self):
        """
        Counts the number of zeros in both the weight and bias masks.

        Returns:
            num_zeros: The number of zeros in the masks.
            num_el: The total number of parameters in the weight and bias tensors.

        NOTE: Sparsity rate = num_zeros / num_el.
        """
        num_zeros_w = 0
        num_el_w = self.weight.numel()
        if self.mask_w is not None:
            num_zeros_w = num_el_w - self.mask_w.sum()

        num_zeros_b = 0
        num_el_b = self.bias.numel() if self.bias is not None else 0
        if self.mask_b is not None:
            num_zeros_b = num_el_b - self.mask_b.sum()

        num_zeros = num_zeros_w + num_zeros_b

        num_el = num_el_w + num_el_b

        return num_zeros, num_el

    def get_sparsity_rate(self) -> float:
        num_zeros, num_el = self.count_zeros()

        return num_zeros / num_el

    def get_prunable_params(self) -> Dict:
        prunable_params = {"weight": self.weight}

        if self.prune_bias:
            prunable_params["bias"] = self.bias

        return prunable_params

    def set_mask(self, mask: Tensor, param_name: str = "weight") -> None:
        if param_name == "weight":
            self.mask_w = mask
        elif param_name == "bias" and self.prune_bias:
            self.mask_b = mask

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Override loading from state dict in order to load masks when present in
        the dictionary.
        """
        mask_w_name = prefix + "mask_w"
        mask_b_name = prefix + "mask_b"

        if mask_w_name in state_dict:
            self.register_buffer("mask_w", state_dict[mask_w_name])

        if mask_b_name in state_dict:
            self.register_buffer("mask_b", state_dict[mask_b_name])

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        return parser
