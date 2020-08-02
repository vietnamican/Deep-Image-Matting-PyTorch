import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d
from torchsummaryX import summary
from .depthwise import Depthwise
from .groupnorm import GroupNorm


class ConvReluBatchnorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
                 with_batchnorm=True,
                 with_relu=True,
                 with_depthwise=False,
                 with_group_norm=False):
        super(ConvReluBatchnorm, self).__init__()
        if with_depthwise:
            self.conv = Depthwise(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, bias=bias)
        else:
            self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.with_relu = False
        if with_relu:
            self.with_relu = True
            self.relu = ReLU()

        self.with_batchnorm = False
        if with_batchnorm:
            if with_group_norm:
                self.with_group_norm = True
                self.batchnorm = GroupNorm(out_channels)
            else:
                self.with_batchnorm = True
                self.batchnorm = BatchNorm2d(out_channels)

        self._init_weight()

    def forward(self, x):
        if self.with_relu:
            if self.with_batchnorm:
                x = self.conv(x)
                x = self.relu(x)
                x = self.batchnorm(x)
            else:
                x = self.conv(x)
                x = self.relu(x)
        else:
            if self.with_batchnorm:
                x = self.conv(x)
                x = self.batchnorm(x)
            else:
                x = self.conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
