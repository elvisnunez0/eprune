"""

VGG for Pytorch. Adapted from:
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

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

LAYER_CFG = {
    11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    16: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    19: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


@MODEL_REGISTRY.register(name="cifar_vgg", class_type="classification")
class CIFARVGG(BaseClassificationModel):
    def __init__(self, cfg: Namespace):
        super().__init__(cfg=cfg)
        self.depth = getattr(cfg, "model.classification.cifar_vgg.depth")
        self.features = self._make_layers(LAYER_CFG[self.depth])

        if self.sparse_model:
            self.linear = self.linear_module(
                512, self.num_classes, sparsity_scale=self.sparsity_scale
            )
        else:
            self.linear = self.linear_module(512, self.num_classes)

        self.load_from_state_dict()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    self.conv_module(in_channels, x, kernel_size=3, padding=1),
                    BatchNorm2d(x),
                    ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--model.classification.cifar_vgg.depth",
            type=int,
            default=16,
            help="The depth of the VGG model. Defaults to 16.",
        )

        return parser
