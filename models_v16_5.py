import torch
import torch.nn as nn
from torchsummaryX import summary

from config import device, im_size
import migrate_model

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
            k_size,
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
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )
        else:
            conv_mod = nn.Conv2d(int(in_channels),
                                int(n_filters),
                                kernel_size=k_size,
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

class _aspp(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_aspp, self).__init__()
        self.cbr_unit = conv2DBatchNormRelu(inplanes, planes, kernel_size,stride=1,padding=padding,dilation=dilation,with_depthwise=True)

        self._init_weight()

    def forward(self, x):
        x = self.cbr_unit(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
class aspp(nn.Module):
    def __init__(self, inplanes):
        super(aspp, self).__init__()
        dilations = [1,2,3]
        self.aspp1 = _aspp(inplanes, 256, 1, padding=0, dilation=1)
        self.aspp2 = _aspp(inplanes, 256, 3, padding=dilations[0], dilation=dilations[0])
        self.aspp3 = _aspp(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp4 = _aspp(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])

        self.cbr_unit_1 = conv2DBatchNormRelu(1024,256,3,stride=1,padding=1,with_depthwise=True)
        self.cbr_unit_2 = conv2DBatchNormRelu(256,512,3,stride=1,padding=1,with_depthwise=True)
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

class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        orig = outputs
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, orig


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        orig = outputs
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, orig

class segnet2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnet2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=1, padding=1)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        orig = outputs
        return orig
    
class segnetUp1(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp1, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = conv2DBatchNormRelu(in_size, out_size, k_size=5, stride=1, padding=2, with_relu=False)

    def forward(self, inputs):
        outputs = self.upsample(inputs)
        outputs = self.conv(outputs)
        return outputs

class adding(nn.Module):
    def __init__(self):
        super(adding, self).__init__()
    def forward(self, a, b):
        adding = a + b
        return adding

class DIMModel(nn.Module):
    def __init__(self, n_classes=1, in_channels=4, is_unpooling=True, pretrain=True):
        super(DIMModel, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling
        self.pretrain = pretrain

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnet2(256, 512)
        self.res = conv2DBatchNormRelu(512,512,k_size=3, stride=1, padding=1, with_relu=True, with_bn=False)
        self.aspp = aspp(512)
        self.adding = adding()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
        self.down5 = segnetDown3(512, 512)

        self.up5 = segnetUp1(512, 512)
        self.up4 = segnetUp1(1024, 512)
        self.up3 = segnetUp1(1024, 256)
        self.up2 = segnetUp1(512, 128)
        self.up1 = segnetUp1(256, 64)
        self.conv1 = conv2DBatchNormRelu(64, n_classes, k_size=3, stride=1, padding=1, with_relu=True, with_bn=True)

        self.sigmoid = nn.Sigmoid()

        # if self.pretrain:
            # import torchvision.models as models
            # vgg16 = models.vgg16()
            # print(vgg16)
        # self.init_vgg16_params()

    def forward(self, inputs):
        # inputs: [N, 4, 320, 320]
        down1, orig1 = self.down1(inputs) # 320
        down2, orig2 = self.down2(down1)  # 160
        down3, orig3 = self.down3(down2)  # 80
        orig4 = self.down4(down3)
        down4 = orig4
        res = self.res(down4)
        aspp = self.aspp(down4)
        adding = self.adding(res, aspp)
        activation = self.relu(adding)
        maxpool, indices_4 = self.maxpool(activation)
        down5, orig5 = self.down5(maxpool) # 20

        up5 = self.up5(down5)                          # 512 20 20
        preup4 = torch.cat((orig5, up5), dim=1)        # 1024 20 20
        up4 = self.up4(preup4)                         # 512 40 40
        preup3 = torch.cat((activation, up4), dim=1)   # 1024 40 40
        up3 = self.up3(preup3)                         # 256 80 80
        preup2 = torch.cat((orig3, up3), dim=1)        # 512 80 80
        up2 = self.up2(preup2)                         # 128 160 160
        preup1 = torch.cat((orig2, up2), dim=1)        # 256 160 160
        up1 = self.up1(preup1)                         # 64 320 320
        x = self.conv1(up1)


        x = torch.squeeze(x, dim=1)  # [N, 1, 320, 320] -> [N, 320, 320]
        x = self.sigmoid(x)

        return x

    # def init_vgg16_params(self):
        # return
        # migrate_model.migrate(self)

class RefinementModel(nn.Module):
    def __init__(self):
        super(RefinementModel, self).__init__()

        self.conv_1 = conv2DBatchNormRelu(4, 64, 3, 1, 1)
        self.conv_2 = conv2DBatchNormRelu(64,64, 3, 1, 1)
        self.conv_3 = conv2DBatchNormRelu(64,64, 3, 1, 1)
        self.conv_4 = conv2DBatchNormRelu(64, 1, 3, 1, 1, with_bn=False, with_relu=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)

        x = torch.squeeze(x, dim=1)
        x = self.sigmoid(x)
        skip = inputs[:,3,:,:]
        x = x + skip

        return x

if __name__ == '__main__':
    model = DIMModel().to(device)
    print(model)
    summary(model, torch.Tensor(4, 4, 320, 320).cuda())
