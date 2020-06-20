""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import device
from torchsummaryX import summary

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout,kernel_size=3,padding=1,stride=1,bias=True,dilation=1,):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, groups=nin, 
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            bias=bias,
                            dilation=dilation, )
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            kernel_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            with_bn=True,
            with_relu=True,
            with_depthwise=False
    ):
        super(conv2DBatchNormRelu, self).__init__()
        if with_depthwise:
            conv_mod = depthwise_separable_conv(int(in_channels),
                             int(n_filters),
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )
        else:
            conv_mod = nn.Conv2d(int(in_channels),
                                int(n_filters),
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                bias=bias,
                                dilation=dilation, )

        if with_bn:
            if with_relu:
                self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True))
            else:
                self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            if with_relu:
                self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))
            else:
                self.cbr_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, 
            in_channels, 
            out_channels,
            k_size=3,
            stride=1,
            padding=1,
            bias=True,
            dilation=1,
            with_bn=True,
            with_relu=True,
            with_depthwise= False):
        super().__init__()
        self.double_conv = nn.Sequential(
            conv2DBatchNormRelu(in_channels, out_channels, k_size, stride,padding,bias,dilation,with_bn,with_relu,with_depthwise),
            conv2DBatchNormRelu(out_channels, out_channels, k_size, stride,padding,bias,dilation,with_bn,with_relu,with_depthwise)
        )

    def forward(self, x):
        return self.double_conv(x)
class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, 
            in_channels, 
            out_channels,
            k_size=3,
            stride=1,
            padding=1,
            bias=True,
            dilation=1,
            with_bn=True,
            with_relu=True,
            with_depthwise= False):
        super().__init__()
        self.double_conv = nn.Sequential(
            conv2DBatchNormRelu(in_channels, out_channels, k_size, stride,padding,bias,dilation,with_bn,with_relu,with_depthwise),
            conv2DBatchNormRelu(out_channels, out_channels, k_size, stride,padding,bias,dilation,with_bn,with_relu,with_depthwise),
            conv2DBatchNormRelu(out_channels, out_channels, k_size, stride,padding,bias,dilation,with_bn,with_relu,with_depthwise)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, isDouble=True):
        super().__init__()
        if isDouble:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                TripleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ASPP(nn.Module):
    def __init__(self, inplanes):
        super(ASPP, self).__init__()
        dilations = [1,2,3]
        self.aspp1 = conv2DBatchNormRelu(inplanes, 256, kernel_size=1,stride=1,padding=0,dilation=1,with_depthwise=False)
        self.aspp2 = conv2DBatchNormRelu(inplanes, 256, kernel_size=3,stride=1,padding=dilations[0],dilation=dilations[0],with_depthwise=False)
        self.aspp3 = conv2DBatchNormRelu(inplanes, 256, kernel_size=3,stride=1,padding=dilations[1],dilation=dilations[1],with_depthwise=False)
        self.aspp4 = conv2DBatchNormRelu(inplanes, 256, kernel_size=3,stride=1,padding=dilations[2],dilation=dilations[2],with_depthwise=False)

        self.cbr_unit_1 = conv2DBatchNormRelu(1024,256,3,stride=1,padding=1,with_depthwise=False)
        self.cbr_unit_2 = conv2DBatchNormRelu(256,512,3,stride=1,padding=1,with_depthwise=False)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.cbr_unit_1(x)
        x = self.cbr_unit_2(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DIMModel(nn.Module):
    def __init__(self, n_classes=1, in_channels=4, is_unpooling=True, pretrain=True):
    # def __init__(self, n_channels, n_classes, bilinear=True):
        super(DIMModel, self).__init__()
        self.n_channels = in_channels
        n_channels =self.n_channels
        self.n_classes = n_classes
        self.bilinear = False
        bilinear = self.bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512, isDouble=False)
        self.post4 = DoubleConv(512, 512)
        self.aspp = ASPP(512)
        self.down4 = Down(512, 512, isDouble=False)
        factor = 2 if bilinear else 1
        self.down5 = Down(512, 1024 // factor, isDouble=False)
        self.up5 = Up(1024, 1024 // factor, bilinear)
        self.up4 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up1 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        post_x4 = self.post4(x4)
        aspp = self.aspp(post_x4)
        adding = aspp + post_x4
        x5 = self.down4(aspp)
        x6 = self.down5(x5)
        x = self.up5(x6, x5)
        x = self.up4(x, adding)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        x = torch.squeeze(x, dim=1)
        x = self.sigmoid(x)
        return x

if __name__ == '__main__':
    model = DIMModel().to(device)
    summary(model, torch.Tensor(1, 4, 320, 320).cuda())