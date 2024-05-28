"""

EfficientNet for PyTorch. Adapted from:
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/efficientnet.py

"""

from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from layers.activation import ReLU
from layers.norm import BatchNorm2d
from models import MODEL_REGISTRY
from models.classification.base_classification_model import \
    BaseClassificationModel

CFG_DICT = {
    "B0": {
        "num_blocks": [1, 2, 2, 3, 3, 4, 1],
        "expansion": [1, 6, 6, 6, 6, 6, 6],
        "out_channels": [16, 24, 40, 80, 112, 192, 320],
        "kernel_size": [3, 3, 5, 3, 5, 5, 3],
        "stride": [1, 2, 2, 2, 1, 2, 1],
        "dropout_rate": 0.2,
        "drop_connect_rate": 0.2,
    }
}


def swish(x):
    return x * x.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    """Squeeze-and-Excitation block with Swish."""

    def __init__(self, in_channels, se_channels, conv_module):
        super(SE, self).__init__()
        self.se1 = conv_module(in_channels, se_channels, kernel_size=1, bias=True)
        self.se2 = conv_module(se_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    """expansion + depthwise + pointwise + squeeze-excitation"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        conv_module,
        expand_ratio=1,
        se_ratio=0.0,
        drop_rate=0.0,
    ):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = conv_module(
            in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = conv_module(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(1 if kernel_size == 3 else 2),
            groups=channels,
            bias=False,
        )
        self.bn2 = BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels, conv_module)

        # Output
        self.conv3 = conv_module(
            channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


@MODEL_REGISTRY.register(name="cifar_efficientnet_b0", class_type="classification")
class EfficientNet(BaseClassificationModel):
    def __init__(self, cfg: Namespace):
        super().__init__(cfg=cfg)
        self.cfg = CFG_DICT["B0"]
        self.conv1 = self.conv_module(
            3, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)

        if self.sparse_model:
            self.linear = self.linear_module(
                self.cfg["out_channels"][-1],
                self.num_classes,
                sparsity_scale=self.sparsity_scale,
            )
        else:
            self.linear = self.linear_module(
                self.cfg["out_channels"][-1], self.num_classes
            )

        self.load_from_state_dict()

    def _make_layers(self, in_channels):
        layers = []
        cfg = [
            self.cfg[k]
            for k in [
                "expansion",
                "out_channels",
                "num_blocks",
                "kernel_size",
                "stride",
            ]
        ]
        b = 0
        blocks = sum(self.cfg["num_blocks"])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg["drop_connect_rate"] * b / blocks
                layers.append(
                    Block(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        self.conv_module,
                        expansion,
                        se_ratio=0.25,
                        drop_rate=drop_rate,
                    )
                )
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg["dropout_rate"]
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out
