import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

logger = logging.getLogger(__name__)



class GradReverse(Function):
    def __init__(self, lambd):
        assert lambd >= 0
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


class Classifer_FixMatch(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0, stochastic=True):
        super(Classifer_FixMatch, self).__init__()

        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor,4096]
        # ------------------------FixMatch part----------------------------------
        # global average pooling and classifier
        #channels[3]
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)




        self.AveragePooling = nn.AdaptiveAvgPool2d((1, 1))



    def forward(self, x):

        out = self.relu(self.bn1(x))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        out = self.fc(out)
        return out









def build_Classifer_FixMatch(depth, widen_factor, dropout, num_classes):
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return Classifer_FixMatch(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes
                      )