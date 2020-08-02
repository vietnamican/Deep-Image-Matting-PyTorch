import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, ReLU, BatchNorm2d, Sequential, Sigmoid
from torchsummaryX import summary

from .block import Block, UpBlock


class DIMModel(nn.Module):
    def __init__(self):
        super(DIMModel, self).__init__()

        self.short1 = Sequential(
            Block(4, 64, 1, 1, 0, with_group_norm=True), Block(64, 64, 1, 1, 0, with_group_norm=True)
        )

        self.down1_conv1 = Block(4, 64, 3, 2, 1, with_depthwise=False, with_group_norm=True)  # 160 32
        self.down1_conv2 = Block(64, 64, 3, 1, 1, with_depthwise=False, with_group_norm=True)  # 160 32
        self.down1_conv3 = Block(64, 64, 3, 1, 1, with_depthwise=False, with_group_norm=True)  # 160 32

        self.short2 = Sequential(
            Block(64, 64, 1, 1, 0, with_group_norm=True), Block(64, 64, 1, 1, 0, with_group_norm=True)
        )

        self.down2_conv1 = Block(64, 128, 3, 2, 1, with_depthwise=False, with_group_norm=True)  # 80
        self.down2_conv2 = Block(128, 128, 3, 1, 1, with_depthwise=False, with_group_norm=True)  # 80
        self.down2_conv3 = Block(128, 128, 3, 1, 1, with_depthwise=False, with_group_norm=True)  # 80

        self.short3 = Sequential(
            Block(128, 128, 1, 1, 0, with_group_norm=True), Block(128, 128, 1, 1, 0, with_group_norm=True)
        )

        self.down3_conv1 = Block(128, 256, 3, 2, 1, with_depthwise=False, with_group_norm=True)  # 40
        self.down3_conv2 = Block(256, 256, 3, 1, 1, with_depthwise=False, with_group_norm=True)  # 40
        self.down3_conv3 = Block(256, 256, 3, 1, 1, with_depthwise=False, with_group_norm=True)  # 40

        self.short4 = Sequential(
            Block(256, 256, 1, 1, 0, with_group_norm=True), Block(256, 256, 1, 1, 0, with_group_norm=True)
        )

        self.down4_conv1 = Block(256, 512, 3, 2, 1, with_depthwise=False, with_group_norm=True)  # 20
        self.down4_conv2 = Block(512, 512, 3, 1, 1, with_depthwise=False, with_group_norm=True)  # 20
        self.down4_conv3 = Block(512, 512, 3, 1, 1, with_depthwise=False, with_group_norm=True)  # 20

        self.short5 = Sequential(
            Block(512, 512, 1, 1, 0, with_group_norm=True), Block(512, 512, 1, 1, 0, with_group_norm=True)
        )

        self.down5_conv1 = Block(512, 1024, 3, 2, 1, with_group_norm=True)  # 10
        self.down5_conv2 = Block(1024, 1024, 3, 1, 1, with_group_norm=True)  # 10
        self.down5_conv3 = Block(1024, 1024, 3, 1, 1, with_group_norm=True)  # 10

        self.up5_conv1 = UpBlock(2, 1024, 512, 3, 1, 1, with_group_norm=True)  # 20 256
        self.up5_conv2 = Block(512, 512, 3, 1, 1, with_group_norm=True)  # 20 256
        self.up5_conv3 = Block(512, 512, 3, 1, 1, with_group_norm=True)  # 20 256

        self.up4_conv1 = UpBlock(2, 512, 256, 3, 1, 1, with_group_norm=True)  # 40 128
        self.up4_conv2 = Block(256, 256, 3, 1, 1, with_group_norm=True)  # 40 128
        self.up4_conv3 = Block(256, 256, 3, 1, 1, with_group_norm=True)  # 40

        self.up3_conv1 = UpBlock(2, 256, 128, 3, 1, 1, with_group_norm=True)  # 80
        self.up3_conv2 = Block(128, 128, 3, 1, 1, with_group_norm=True)  # 80
        self.up3_conv3 = Block(128, 128, 3, 1, 1, with_group_norm=True)  # 80

        self.up2_conv1 = UpBlock(2, 128, 64, 3, 1, 1, with_group_norm=True)  # 160
        self.up2_conv2 = Block(64, 64, 3, 1, 1, with_group_norm=True)  # 160
        self.up2_conv3 = Block(64, 64, 3, 1, 1, with_group_norm=True)  # 160

        self.up1_conv1 = UpBlock(2, 64, 64, 3, 1, 1, use_transpose_conv=True, with_group_norm=True)  # 320
        self.up1_conv2 = Block(64, 64, 3, 1, 1, with_group_norm=True)  # 320
        self.up1_conv3 = Block(64, 64, 3, 1, 1, with_group_norm=True)  # 320

        self.lastconv = Sequential(
            Block(64, 128, 3, 1, 1, with_group_norm=True),
            Block(128, 128, 3, 1, 1, with_group_norm=True),
            Block(128, 1, 1, 1, 0, with_group_norm=True)
        )

        self.sigmoid = Sigmoid()

    def forward(self, x):
        s1 = self.short1(x)

        x = self.down1_conv1(x)
        x = self.down1_conv2(x)
        x = self.down1_conv3(x)

        s2 = self.short2(x)

        x = self.down2_conv1(x)
        x = self.down2_conv2(x)
        x = self.down2_conv3(x)

        s3 = self.short3(x)

        x = self.down3_conv1(x)
        x = self.down3_conv2(x)
        x = self.down3_conv3(x)

        s4 = self.short4(x)

        x = self.down4_conv1(x)
        x = self.down4_conv2(x)
        x = self.down4_conv3(x)

        s5 = self.short5(x)

        x = self.down5_conv1(x)
        x = self.down5_conv2(x)
        x = self.down5_conv3(x)

        x = self.up5_conv1(x, s5.shape[2:4])
        x = self.up5_conv2(x)
        x = self.up5_conv3(x)

        x = x + s5

        x = self.up4_conv1(x, s4.shape[2:4])
        x = self.up4_conv2(x)
        x = self.up4_conv3(x)

        x = x + s4

        x = self.up3_conv1(x, s3.shape[2:4])
        x = self.up3_conv2(x)
        x = self.up3_conv3(x)

        x = x + s3

        x = self.up2_conv1(x, s2.shape[2:4])
        x = self.up2_conv2(x)
        x = self.up2_conv3(x)

        x = x + s2

        x = self.up1_conv1(x, s1.shape[2:4])
        x = self.up1_conv2(x)
        x = self.up1_conv3(x)

        x = x + s1

        x = self.lastconv(x)
        x = torch.squeeze(x, dim=1)
        x = self.sigmoid(x)

        return x
