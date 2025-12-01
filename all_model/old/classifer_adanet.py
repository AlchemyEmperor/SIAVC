import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

logger = logging.getLogger(__name__)



class identical(nn.Module):

    def forward(self, input):
        return input

# class GradReverse(Function):
#     def __init__(self, lambd):
#         assert lambd >= 0
#         self.lambd = lambd
#
#     def forward(self, x):
#         return x.view_as(x)
#
#     def backward(self, grad_output):
#         return (grad_output * -self.lambd)



class Classifer_Adanet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0, stochastic=True, top_bn=False, lambd=0.0):
        super(Classifer_Adanet, self).__init__()
        self.top_bn = top_bn
        self.lambd = float(lambd)


        # ------------------Adanet parts------------------------
        self.AveragePooling = nn.AdaptiveAvgPool2d((1, 1))

        # ------------------全连接层(adanet 分类器)------------------------
        self.fc = nn.Linear(128, num_classes)

        self.top_bn_layer = nn.BatchNorm1d(num_classes)
        dropout = nn.Dropout2d() if stochastic else identical()

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            dropout
        )

        # -----------------鉴别器(adanet 鉴别器)--------------------------
        self.discriminator = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2),
            nn.Softmax(dim=1)
        )


    def forward(self, x):


        # Classifier Part (interpolated)
        feature = self.AveragePooling(x)
        feature = feature.view(feature.shape[0], -1)
        # classifier branch
        out = self.fc(feature)
        if self.top_bn:
            out = self.top_bn_layer(out)
        out = F.softmax(out, 1)
        # discriminator branch
        #cls = self.discriminator(GradReverse(self.lambd)(feature))
        cls = self.discriminator(feature)
        return out, cls







def build_Classifer_Adanet(depth, widen_factor, dropout, num_classes):
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return Classifer_Adanet(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes
                      )