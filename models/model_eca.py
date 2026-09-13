import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MobileNetV2


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ResidualECAGate(nn.Module):
    """ECA channel gate with an identity-preserving learnable residual scale."""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.res_scale = nn.Parameter(torch.zeros(1))
        self.kernel_size = kernel_size

    def forward(self, x):
        weights = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights).transpose(-1, -2).unsqueeze(-1)
        weights = self.sigmoid(weights)
        return x * (1.0 + self.res_scale * weights)


class NeighborFeatureAggregationECA(nn.Module):
    def __init__(self, in_d):
        super().__init__()
        self.in_d = in_d
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.eca = nn.ModuleList([ResidualECAGate(channels) for channels in in_d])
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(in_d[0], in_d[1], 3, padding=1),
            nn.BatchNorm2d(in_d[1]),
            nn.ReLU(inplace=True),
        )
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(in_d[1], in_d[1], 3, padding=1),
            nn.BatchNorm2d(in_d[1]),
            nn.ReLU(inplace=True),
        )
        self.conv2d3 = nn.Sequential(
            nn.Conv2d(in_d[2], in_d[2], 3, padding=1),
            nn.BatchNorm2d(in_d[2]),
            nn.ReLU(inplace=True),
        )
        self.conv2d4 = nn.Sequential(
            nn.Conv2d(in_d[3], in_d[3], 3, padding=1),
            nn.BatchNorm2d(in_d[3]),
            nn.ReLU(inplace=True),
        )
        self.double_conv5 = DoubleConv(in_d[4], in_d[3])
        self.cat4 = DoubleConv(2 * in_d[3], in_d[2], in_d[3])
        self.cat3 = DoubleConv(2 * in_d[2], in_d[1], in_d[2])
        self.cat2 = DoubleConv(3 * in_d[1], in_d[1], 2 * in_d[1])
        self.cls = nn.Conv2d(in_d[1], 1, 3, padding=1)
        self.aux3 = nn.Conv2d(in_d[1], 1, 1)
        self.aux4 = nn.Conv2d(in_d[2], 1, 1)
        self.aux5 = nn.Conv2d(in_d[3], 1, 1)

    def forward(self, *features):
        x1 = features[:5]
        x2 = features[5:]
        differences = [torch.abs(a - b) for a, b in zip(x1, x2)]
        c1, c2, c3, c4, c5 = [gate(diff) for gate, diff in zip(self.eca, differences)]

        d1 = self.conv2d1(self.downsample(c1))
        d2 = self.conv2d2(c2)
        d3 = self.conv2d3(c3)
        d4 = self.conv2d4(c4)
        d5 = self.double_conv5(c5)

        e4 = F.interpolate(d5, size=d4.shape[-2:], mode='bilinear', align_corners=False)
        f4 = self.cat4(torch.cat([d4, e4], dim=1))
        e3 = F.interpolate(f4, size=d3.shape[-2:], mode='bilinear', align_corners=False)
        f3 = self.cat3(torch.cat([d3, e3], dim=1))
        e2 = F.interpolate(f3, size=d2.shape[-2:], mode='bilinear', align_corners=False)
        s2 = self.cat2(torch.cat([e2, d2, d1], dim=1))

        return self.cls(s2), self.aux3(f3), self.aux4(f4), self.aux5(d5)


class BaseNetECA(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, pretrained=True):
        super().__init__()
        self.backbone = MobileNetV2.mobilenet_v2(pretrained=pretrained)
        self.swa = NeighborFeatureAggregationECA([16, 24, 32, 96, 320])

    def forward(self, x1, x2):
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        logits = self.swa(*(f1 + f2))
        target_size = x1.shape[-2:]
        return tuple(
            F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            for x in logits
        )
