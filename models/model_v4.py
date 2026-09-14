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


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SpatialDifferenceGate(nn.Module):
    def __init__(self, kernel_size=7, initial_scale=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, difference):
        context = torch.cat(
            [difference.mean(dim=1, keepdim=True), difference.amax(dim=1, keepdim=True)], dim=1
        )
        attention = torch.sigmoid(self.spatial(context))
        return difference * (1.0 + self.scale * attention)


class NeighborFeatureAggregationV4(nn.Module):
    def __init__(self, in_d):
        super().__init__()
        self.in_d = in_d
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.spatial_gate_c2 = SpatialDifferenceGate(kernel_size=7, initial_scale=0.1)
        self.spatial_gate_c3 = SpatialDifferenceGate(kernel_size=7, initial_scale=0.1)
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(in_d[0], in_d[1], 3, padding=1),
            nn.BatchNorm2d(in_d[1]), nn.ReLU(inplace=True),
        )
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(in_d[1], in_d[1], 3, padding=1),
            nn.BatchNorm2d(in_d[1]), nn.ReLU(inplace=True),
        )
        self.conv2d3 = nn.Sequential(
            nn.Conv2d(in_d[2], in_d[2], 3, padding=1),
            nn.BatchNorm2d(in_d[2]), nn.ReLU(inplace=True),
        )
        self.conv2d4 = nn.Sequential(
            nn.Conv2d(in_d[3], in_d[3], 3, padding=1),
            nn.BatchNorm2d(in_d[3]), nn.ReLU(inplace=True),
        )
        self.double_conv5 = DoubleConv(in_d[4], in_d[3])
        self.cat4 = DoubleConv(2 * in_d[3], in_d[2], in_d[3])
        self.cat3 = DoubleConv(2 * in_d[2], in_d[1], in_d[2])
        self.cat2 = DoubleConv(3 * in_d[1], in_d[1], 2 * in_d[1])

        self.shallow_boundary = DepthwiseSeparableConv(2 * in_d[0], in_d[0])
        self.refine_half = DepthwiseSeparableConv(in_d[1] + in_d[0], in_d[1])
        self.refine_full = DepthwiseSeparableConv(in_d[1], in_d[0])
        self.boundary_scale = nn.Parameter(torch.tensor(0.1))
        self.detail_scale = nn.Parameter(torch.tensor(0.1))
        self.cls = nn.Conv2d(in_d[1], 1, 3, padding=1)
        self.detail_cls = nn.Conv2d(in_d[0], 1, 3, padding=1)
        self.aux3 = nn.Conv2d(in_d[1], 1, 1)
        self.aux4 = nn.Conv2d(in_d[2], 1, 1)
        self.aux5 = nn.Conv2d(in_d[3], 1, 1)

    def forward(self, *features):
        x1 = features[:5]
        x2 = features[5:]
        c1, c2, c3, c4, c5 = [torch.abs(a - b) for a, b in zip(x1, x2)]
        c2 = self.spatial_gate_c2(c2)
        c3 = self.spatial_gate_c3(c3)
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

        c1_high_frequency = torch.abs(c1 - F.avg_pool2d(c1, 3, stride=1, padding=1))
        shallow = self.shallow_boundary(torch.cat([c1, c1_high_frequency], dim=1))
        up_half = F.interpolate(s2, size=shallow.shape[-2:], mode='bilinear', align_corners=False)
        half_correction = self.refine_half(torch.cat([up_half, shallow], dim=1))
        refined_half = up_half + self.boundary_scale * half_correction
        refined_full = self.refine_full(
            F.interpolate(refined_half, scale_factor=2.0, mode='bilinear', align_corners=False)
        )
        coarse = F.interpolate(self.cls(s2), size=refined_full.shape[-2:], mode='bilinear', align_corners=False)
        main = coarse + self.detail_scale * self.detail_cls(refined_full)
        return main, self.aux3(f3), self.aux4(f4), self.aux5(d5)


class BaseNetV4(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, pretrained=True):
        super().__init__()
        self.backbone = MobileNetV2.mobilenet_v2(pretrained=pretrained)
        self.swa = NeighborFeatureAggregationV4([16, 24, 32, 96, 320])

    def forward(self, x1, x2):
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        logits = self.swa(*(f1 + f2))
        target_size = x1.shape[-2:]
        return tuple(
            x if x.shape[-2:] == target_size else F.interpolate(
                x, size=target_size, mode='bilinear', align_corners=False
            ) for x in logits
        )
