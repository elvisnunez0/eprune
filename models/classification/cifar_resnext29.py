"""

ResNeXt for PyTorch. Adapted from:
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnext.py

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


class Block(nn.Module):
    """Grouped convolution block."""

    expansion = 2

    def __init__(
        self, in_planes, conv_module, cardinality=32, bottleneck_width=4, stride=1
    ):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = conv_module(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(group_width)
        self.conv2 = conv_module(
            group_width,
            group_width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = BatchNorm2d(group_width)
        self.conv3 = conv_module(
            group_width, self.expansion * group_width, kernel_size=1, bias=False
        )
        self.bn3 = BatchNorm2d(self.expansion * group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(
                conv_module(
                    in_planes,
                    self.expansion * group_width,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(self.expansion * group_width),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


@MODEL_REGISTRY.register(name="cifar_resnext29", class_type="classification")
class ResNeXt(BaseClassificationModel):

    def __init__(
        self,
        cfg: Namespace,
    ):
        super().__init__(cfg=cfg)

        self.num_blocks = [3, 3, 3]
        self.cardinality = getattr(
            cfg, "model.classification.cifar_resnext29.cardinality"
        )
        self.running_bottleneck_width = getattr(
            cfg, "model.classification.cifar_resnext29.bottleneck_width"
        )
        self.bottleneck_width = self.running_bottleneck_width
        self.in_planes = 64

        self.conv1 = self.conv_module(3, 64, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(self.num_blocks[0], 1)
        self.layer2 = self._make_layer(self.num_blocks[1], 2)
        self.layer3 = self._make_layer(self.num_blocks[2], 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        if self.sparse_model:
            self.linear = self.linear_module(
                self.cardinality * self.bottleneck_width * 8,
                self.num_classes,
                sparsity_scale=self.sparsity_scale,
            )
        else:
            self.linear = self.linear_module(
                self.cardinality * self.bottleneck_width * 8, self.num_classes
            )

        self.load_from_state_dict()

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                Block(
                    self.in_planes,
                    self.conv_module,
                    self.cardinality,
                    self.running_bottleneck_width,
                    stride,
                )
            )
            self.in_planes = (
                Block.expansion * self.cardinality * self.running_bottleneck_width
            )
        # Increase bottleneck_width by 2 after each stage.
        self.running_bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--model.classification.cifar-resnext29.cardinality",
            type=int,
            default=2,
            help="The cardinality of the ResNext29 model. Defaults to 2.",
        )

        group.add_argument(
            "--model.classification.cifar-resnext29.bottleneck-width",
            type=int,
            default=64,
            choices=[4, 64],
            help="The width of bottleneck layers. Defaults to 64.",
        )

        return parser
