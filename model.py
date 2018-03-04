import torch
import torch.nn as nn


def conv(c_in, c_out, k, s, p):
    return nn.Conv2d(
            c_in, c_out, kernel_size=k,
            stride=s, padding=p)


def conv3x3(c_in, c_out, stride=1, pad=1):
    return conv(c_in, c_out, 3, stride, pad)


def conv1x1(c_in, c_out, stride=1, pad=0):
    return conv(c_in, c_out, 1, stride, pad)


def ConvBlock(c_in, c_out):
    return nn.Sequential(
            nn.InstanceNorm2d(c_in),
            nn.LeakyReLU(),
            conv3x3(c_in, c_out),
            nn.InstanceNorm2d(c_out),
            nn.LeakyReLU(),
            conv3x3(c_out, c_out))


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.squeeze(x)


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = ConvBlock(channels, channels)

    def forward(self, x):
        return self.conv(x) + x


def UpBlock(c_in, c_out):
    return nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(c_in, c_out))


def DownBlock(c_in, c_out):
    return nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlock(c_in, c_out))


def BasicNetwork(c_in, num_classes):
    return nn.Sequential(
            conv3x3(c_in, 32),
            DownBlock(32, 64),
            DownBlock(64, 128),
            ResNetBlock(128),
            ResNetBlock(128),
            nn.AdaptiveAvgPool2d(1),
            Squeeze(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes))
