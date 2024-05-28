import torch
from torch import Tensor


def patchify(x: Tensor, p: int, flatten: bool = True) -> Tensor:
    """
    Given a batch of images with shape [B, C, H, W] and a
    patch size p, patchifies x in raster order.

    Args:
        x: A tensor representing a batch of images with shape [B, C, H, W].
        p: An integer specifying the patch size.
        flatten: If true, will flatten the pixel/channel dimensions.

    Returns:
        x_patchified: A tensor representing the patchified x.
            If @flatten=True, the returned shape is:
                [B, num_patches, C*p*p] where num_patches = (H // p) * (W // p).
            If @flatten=False, the returned shape is:
                [B, num_patches, C, p, p] where num_patches = (H // p) * (W // p).
    """
    B, C, H, W = x.shape
    num_patches_h = H // p
    num_patches_w = W // p

    x_patchified = x.reshape((B, C, num_patches_h, p, num_patches_w, p))
    x_patchified = torch.einsum("bchpwq->bhwpqc", x_patchified)

    if flatten:
        x_patchified = x_patchified.reshape(
            (B, num_patches_h * num_patches_w, C * p * p)
        )
    else:
        x_patchified = x_patchified.reshape((B, num_patches_h * num_patches_w, C, p, p))

    return x_patchified


def total_variation_patchwise(x: Tensor) -> Tensor:
    """
    Computes the total variation of a patchified batch of images for each patch.

    Args:
        x: A batch of patchified images with shape [B, num_patches, C, patch_size, patch_size].

    Returns:
        tv: A tensor with shape [B, num_patches] where tv[i,j] represents the
            total variation of patch j of image i.
    """

    diff_h = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    diff_w = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

    sum_dim = [2, 3, 4]
    tv = torch.sum(torch.abs(diff_h), dim=sum_dim) + torch.sum(
        torch.abs(diff_w), dim=sum_dim
    )

    return tv
