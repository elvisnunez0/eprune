"""

ResNet for CIFAR. Adapted from:
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

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

DEPTH_LAYER_CFG = {
    18: ("basic", [2, 2, 2, 2]),
    34: ("basic", [3, 4, 6, 3]),
    50: ("bottleneck", [3, 4, 6, 3]),
    101: ("bottleneck", [3, 4, 23, 3]),
    152: ("bottleneck", [3, 8, 36, 3]),
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_module, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_module(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv_module(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(planes)
        self.relu = ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_module(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, conv_module, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_module(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv_module(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = conv_module(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = BatchNorm2d(self.expansion * planes)
        self.relu = ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_module(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


@MODEL_REGISTRY.register(name="cifar_resnet", class_type="classification")
class CIFARResNet(BaseClassificationModel):
    def __init__(self, cfg: Namespace):
        super().__init__(cfg=cfg)

        self.depth = getattr(cfg, "model.classification.cifar_resnet.depth")
        block, num_blocks = DEPTH_LAYER_CFG[self.depth]
        block = BasicBlock if block == "basic" else Bottleneck
        self.in_planes = 64

        self.conv1 = self.conv_module(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        if self.sparse_model:
            self.linear = self.linear_module(
                512 * block.expansion,
                self.num_classes,
                sparsity_scale=self.sparsity_scale,
            )
        else:
            self.linear = self.linear_module(512 * block.expansion, self.num_classes)

        self.relu = ReLU()

        self.load_from_state_dict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes, planes, conv_module=self.conv_module, stride=stride
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        return out

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--model.classification.cifar_resnet.depth",
            type=int,
            default=18,
            help="The depth of the ResNet model. Defaults to 18.",
        )

        return parser
