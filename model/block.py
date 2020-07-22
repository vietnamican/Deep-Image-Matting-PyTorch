import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d, UpsamplingNearest2d, ConvTranspose2d
from torchsummaryX import summary

from .conv_relu_bn import ConvReluBatchnorm
from .depthwise import Depthwise


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, with_depthwise=True,
                 grow_first=True):
        super(Block, self).__init__()

        if grow_first:
            self.crb1 = ConvReluBatchnorm(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, with_depthwise=with_depthwise)
            self.crb2 = ConvReluBatchnorm(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=1, padding=padding, with_depthwise=with_depthwise)
        else:
            self.crb1 = ConvReluBatchnorm(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                          stride=1, padding=padding, with_depthwise=with_depthwise)
            self.crb2 = ConvReluBatchnorm(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=1, padding=padding, with_depthwise=with_depthwise)

        if with_depthwise:
            self.conv = Depthwise(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=1, padding=padding)
        else:
            self.conv = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding)

        self.skip = Depthwise(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                              padding=0)
        self.relu = ReLU()
        self.batchnorm = BatchNorm2d(out_channels)

    def forward(self, x):
        skip = self.skip(x)

        x = self.crb1(x)
        x = self.crb2(x)
        x = self.conv(x)

        x = x + skip

        x = self.relu(x)
        x = self.batchnorm(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)


class UpBlock(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size=3, stride=1, padding=1, with_depthwise=True,
                 grow_first=True, use_transpose_conv=False):
        super(UpBlock, self).__init__()
        if use_transpose_conv:
            self.up = ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.up = UpsamplingNearest2d(scale_factor=scale_factor)

        self.block = Block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, with_depthwise=with_depthwise, grow_first=grow_first)

    def forward(self, x, size):
        x = self.up(x)
        original_h, original_w = size
        output_h, output_w = x.shape[2:4]
        diff_h = original_h - output_h
        diff_w = original_w - output_w
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])
        x = self.block(x)

        return x
