"""
DenseNet for PyTorch. Adapted from:
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py

"""

import math
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

DEPTH_CFG = {
    "121c": {"num_blocks": [6, 12, 24, 16], "growth_rate": 12},
    "121": {"num_blocks": [6, 12, 24, 16], "growth_rate": 32},
    "169": {"num_blocks": [6, 12, 32, 32], "growth_rate": 32},
    "201": {"num_blocks": [6, 12, 48, 32], "growth_rate": 32},
    "161": {"num_blocks": [6, 12, 36, 24], "growth_rate": 48},
}


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, conv_module):
        super(Bottleneck, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.conv1 = conv_module(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = BatchNorm2d(4 * growth_rate)
        self.conv2 = conv_module(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):

    def __init__(self, in_planes, out_planes, conv_module):
        super(Transition, self).__init__()
        self.bn = BatchNorm2d(in_planes)
        self.conv = conv_module(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


@MODEL_REGISTRY.register(name="cifar_densenet", class_type="classification")
class DenseNet(BaseClassificationModel):
    def __init__(self, cfg: Namespace):
        super().__init__(cfg=cfg)

        self.depth_cfg = getattr(cfg, "model.classification.cifar_densenet.depth_cfg")
        self.growth_rate = DEPTH_CFG[self.depth_cfg]["growth_rate"]
        self.num_blocks = DEPTH_CFG[self.depth_cfg]["num_blocks"]
        self.reduction = 0.5
        block = Bottleneck

        num_planes = 2 * self.growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, self.num_blocks[0])
        num_planes += self.num_blocks[0] * self.growth_rate
        out_planes = int(math.floor(num_planes * self.reduction))
        self.trans1 = Transition(num_planes, out_planes, self.conv_module)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, self.num_blocks[1])
        num_planes += self.num_blocks[1] * self.growth_rate
        out_planes = int(math.floor(num_planes * self.reduction))
        self.trans2 = Transition(num_planes, out_planes, self.conv_module)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, self.num_blocks[2])
        num_planes += self.num_blocks[2] * self.growth_rate
        out_planes = int(math.floor(num_planes * self.reduction))
        self.trans3 = Transition(num_planes, out_planes, self.conv_module)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, self.num_blocks[3])
        num_planes += self.num_blocks[3] * self.growth_rate

        self.bn = nn.BatchNorm2d(num_planes)

        if self.sparse_model:
            self.linear = self.linear_module(
                num_planes, self.num_classes, sparsity_scale=self.sparsity_scale
            )
        else:
            self.linear = self.linear_module(num_planes, self.num_classes)

        self.load_from_state_dict()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, self.conv_module))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--model.classification.cifar_densenet.depth_cfg",
            type=str,
            default="121c",
            choices=["121c", "121", "161", "169", "201"],
            help="The depth configuration of the DenseNet model. Defaults to 18.",
        )

        return parser
