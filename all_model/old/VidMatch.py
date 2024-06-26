from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

def convert(x):
    x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
    return x

class identical(nn.Module):

    def forward(self, input):
        return input

class GradReverse(Function):
    def __init__(self, lambd):
        assert lambd >= 0
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ThreeD_CNN(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0):
        super().__init__()
        # 3D CNN branch
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x






class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 stochastic=True,
                 top_bn=False,
                 n_classes=7,
                 lambd=0.0):
        super().__init__()

        # 3D CNN branch
        self.backbone = ThreeD_CNN(block, layers, block_inplanes)
        channels = [16, 32, 64, 128, 512, 1024, 2048, 4096]
        # ------------------------FixMatch part----------------------------------
        # global average pooling and classifier
        # channels[3]
        self.bn1_f = nn.BatchNorm2d(channels[4], momentum=0.001)
        self.bn2_f = nn.LayerNorm(channels[4])
        self.relu_f = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.fc_f = nn.Linear(channels[4], n_classes)
        self.fc_f = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.channels_f = channels[4]

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight,
        #                                 mode='fan_out',
        #                                 nonlinearity='leaky_relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1.0)
        #         nn.init.constant_(m.bias, 0.0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0.0)

        self.AveragePooling = nn.AdaptiveAvgPool2d((1, 1))


        # ------------------Adanet parts------------------------
        self.top_bn_a = top_bn
        self.lambd_a = float(lambd)
        # self.AveragePooling_a = nn.AdaptiveAvgPool2d((1, 1))

        # ------------------------------------------
        self.fc_a = nn.Linear(512, n_classes)

        self.top_bn_layer_a = nn.BatchNorm1d(n_classes)
        # dropout = nn.Dropout2d() if stochastic else identical()

        # self.bottleneck_a = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        #     dropout
        # )

        # -----------------------------------------
        self.discriminator_a = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x, mode):

        x = self.backbone(x) #4 512 2 5 5


        # x_f = convert(x) #4 1024 5 5
        # x_f = self.relu_f(self.bn1_f(x_f))
        # x_f = F.adaptive_avg_pool2d(x_f, 1)
        # x_f = x_f.view(-1, self.channels_f)
        # x_f = self.fc_f(x_f)
        x_f = x.view(x.size(0), -1)
        x_f = self.fc_f(x_f)


        # x_a = convert(x)
        feature = x.view(x.size(0), -1)
        # Classifier Part (interpolated)
        # feature = self.AveragePooling_a(x_a)
        # feature = feature.view(feature.shape[0], -1)
        # classifier branch
        out = self.fc_a(feature)
        if self.top_bn_a:
            out = self.top_bn_layer_a(out)
        out = F.softmax(out, 1)  # (2,2)
        # discriminator branch

        # feature.view_as(feature)  # (2,2048)
        # cls = self.discriminator(GradReverse(self.lambd).forward(feature))
        cls = self.discriminator_a(feature)

        return x_f, out, cls


    # def forward(self, x, mode):
    #
    #     x = self.backbone(x)
    #     if mode == 'F':
    #         # x = torch.squeeze(x, -1)
    #         x = convert(x)
    #         x = self.relu_f(self.bn1_f(x))
    #         x = F.adaptive_avg_pool2d(x, 1)
    #         x = x.view(-1, self.channels_f)
    #         x = self.fc_f(x)
    #
    #         return x
    #
    #     if mode == 'A':
    #         x = convert(x)
    #         # Classifier Part (interpolated)
    #         feature = self.AveragePooling_a(x)
    #         feature = feature.view(feature.shape[0], -1)
    #         # classifier branch
    #         out = self.fc_a(feature)
    #         if self.top_bn_a:
    #             out = self.top_bn_layer_a(out)
    #         out = F.softmax(out, 1)  # (2,2)
    #         # discriminator branch
    #
    #         feature.view_as(feature)  # (2,2048)
    #         # cls = self.discriminator(GradReverse(self.lambd).forward(feature))
    #         cls = self.discriminator_a(feature)
    #
    #         return out, cls



def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model