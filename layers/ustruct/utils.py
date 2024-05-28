from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def magnitude_threshold(
    tensor: Tensor, target_sparsity: float, prev_mask: Tensor = None
) -> Tuple[float, Tensor]:
    """
    Given a tensor, returns the kth smallest value of the tensor by magnitude
    where k = round(tensor.numel() * target_sparsity).

    Args:
        tensor: A tensor of arbitrary shape (will be flattened).
        target_sparsity: A value between 0 and 1 representing the number
            of 0s the provided tensor should have. The number of zeros is
            the number of elements in the tensor multiplied by this value.

    Returns:
        threshold: The kth smallest value (by magnitude) in @tensor.
    """
    if target_sparsity == 0:
        threshold = -torch.inf
    else:
        values = tensor.flatten()
        if prev_mask is not None:
            values = values[prev_mask.flatten()]

        importance = values.abs()
        num_zero = round(values.numel() * target_sparsity)
        threshold, _ = torch.kthvalue(importance, k=num_zero)

    return threshold


def magnitude_prune_mask(
    tensor: Tensor,
    target_sparsity: Optional[float] = None,
    threshold: Optional[float] = None,
    prev_mask: Tensor = None,
) -> Tuple[float, Tensor]:
    """
    Given a tensor, returns a binary mask with the same shape. Values of 1 indicate
    the value is >= the threshold; 0 indicates less than. If the threshold is not
    provided, it is computed as the kth smallest value of the tensor by magnitude
    where k = round(tensor.numel() * target_sparsity).

    NOTE: If the threshold if specified, then @target_sparsity will be ignored and vice
        versa. Hence AT LEAST ONE OF @target_sparsity and @threshold should be
        specified. If both are, the specified threshold will be used.

    Args:
        tensor: A tensor of arbitrary shape (will be flattened).
        target_sparsity: A value between 0 and 1 representing the number
            of 0s the provided tensor should have. The number of zeros is
            the number of elements in the tensor multiplied by this value.
        threshold: The value above which indices in the mask will be 1 if
            the corresponding element is larger than this value.


    Returns:
        mask: A binary tensor with the same shape as @tensor.
    """
    # At least one of @target_sparsity and @threshold must be specified.
    assert 1 <= (target_sparsity is None) + (threshold is None) < 2

    if threshold is None:
        threshold = magnitude_threshold(
            tensor=tensor, target_sparsity=target_sparsity, prev_mask=prev_mask
        )

    param_importance = tensor.abs()
    mask = torch.gt(param_importance, threshold)

    if prev_mask is not None:
        mask = mask * prev_mask

    return mask


def rand_mask_like(
    tensor: Tensor, fraction_zeros: float, device: torch.device
) -> Tensor:
    """
    Given a tensor, constructs a binary tensor of the same shape with the
    specified fraction of zeros.

    Args:
        tensor: A tensor whose shape will be matches.
        fraction_zeros: A float between 0 and 1 specifying the fraction of
            zeros in the returned tensor.
        device: The deveice to move the returned tensor to.

    Returns:
        mask: A random binary tensor with the same shape as @tensor. The fraction
            of zeros is @fraction_zeros.
    """
    shape = tensor.shape
    num_el = tensor.numel()

    num_zeros = round(num_el * fraction_zeros)
    num_ones = num_el - num_zeros

    mask = torch.cat([torch.zeros(num_zeros), torch.ones(num_ones)])
    mask = mask[torch.randperm(num_el)]
    mask = mask.reshape(shape)
    mask = mask.to(device)

    return mask


def get_global_params(model: nn.Module, prune_conv: bool, prune_linear: bool) -> Tensor:
    """
    Returns all of the parameters of a model flattened in a 1-d tensor.

    NOTE: Normalization parameters are not considered, only conv/linear.

    Args:
        - model: The model whose parameters will be extracted.
        - include_conv: Whether to include convolution layer parameters.
        - include_linear: Whether to include linear layer parameters.
    Returns:
        params: The flattened tensor of parameters.
    """
    params = torch.tensor([])

    for module in model.modules():
        if (isinstance(module, nn.Conv2d) and prune_conv) or (
            isinstance(module, nn.Linear) and prune_linear
        ):
            params = torch.cat([params, module.weight.detach().flatten().cpu()])

    return params


def get_global_grad2_accs(
    model: nn.Module, prune_conv: bool, prune_linear: bool
) -> Tensor:
    """
    Returns the grad2_acc of all the model's parameters in a 1-d tensor.

    NOTE: Normalization parameters are not considered, only conv/linear weights.

    Args:
        - model: The model whose parameters will be extracted.
        - include_conv: Whether to include convolution layer parameters.
        - include_linear: Whether to include linear layer parameters.
    Returns:
        params: The flattened tensor of the model's parameters' grad2_acc attribute.
    """
    params = torch.tensor([])

    for module in model.modules():
        if (isinstance(module, nn.Conv2d) and prune_conv) or (
            isinstance(module, nn.Linear) and prune_linear
        ):
            # Only consider weight param
            grad2_acc = module.weight.grad2_acc.detach().flatten().cpu()
            params = torch.cat([params, grad2_acc])

    return params
