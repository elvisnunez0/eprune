"""

All-CNN adapted from:
    https://github.com/StefOe/all-conv-pytorch/blob/master/allconv.py

"""

from argparse import Namespace

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from layers.activation import ReLU
from layers.conv import Conv2d
from models import MODEL_REGISTRY
from models.classification.base_classification_model import \
    BaseClassificationModel


@MODEL_REGISTRY.register(name="allcnn", class_type="classification")
class AllConvNet(BaseClassificationModel):
    def __init__(self, cfg: Namespace, *args, **kwargs):
        super().__init__(cfg=cfg)

        self.conv1 = Conv2d(3, 96, 3, padding=1)
        self.conv2 = Conv2d(96, 96, 3, padding=1)
        self.conv3 = Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = Conv2d(96, 192, 3, padding=1)
        self.conv5 = Conv2d(192, 192, 3, padding=1)
        self.conv6 = Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = Conv2d(192, 192, 3, padding=1)
        self.conv8 = Conv2d(192, 192, 1)

        self.relu = ReLU()

        self.class_conv = Conv2d(192, self.num_classes, 1)

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout:
            x = F.dropout(x, 0.2)

        conv1_out = self.relu(self.conv1(x))
        conv2_out = self.relu(self.conv2(conv1_out))
        conv3_out = self.relu(self.conv3(conv2_out))

        if self.dropout:
            conv3_out = F.dropout(conv3_out, 0.5)

        conv4_out = self.relu(self.conv4(conv3_out))
        conv5_out = self.relu(self.conv5(conv4_out))
        conv6_out = self.relu(self.conv6(conv5_out))

        if self.dropout:
            conv6_out = F.dropout(conv6_out, 0.5)

        conv7_out = self.relu(self.conv7(conv6_out))
        conv8_out = self.relu(self.conv8(conv7_out))

        class_out = self.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)  # [B, num_classes, 1, 1]

        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)

        return pool_out

    def initialize(self):
        for module in self.modules():
            if isinstance(module, Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                nn.init.constant_(module.bias, val=0.0)
