import torch
import torch.nn as nn


def conv(c_in, c_out, k, s, p):
    return nn.Conv2d(
            c_in, c_out, kernel_size=k,
            stride=s, padding=p)

def conv3x3(c_in, c_out, downsample=False):
    stride = 1 if not downsample else 2

    return conv(c_in, c_out, 3, stride, 1)


def conv1x1(c_in, c_out, downsample=False):
    stride = 1 if not downsample else 2

    return conv(c_in, c_out, 1, stride, 0)


def ConvBlock(c_in, c_out, downsample=False):
    return nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.LeakyReLU(),
            conv3x3(c_in, c_out, downsample),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(),
            conv3x3(c_out, c_out))


class Squeeze(nn.Module):
    def forward(self, x):
        return torch.squeeze(x)


class Identity(nn.Module):
    def forward(self, x):
        return x


class ResNetBlock(nn.Module):
    def __init__(self, c_in, c_out, downsample=False):
        super().__init__()

        self.conv = ConvBlock(c_in, c_out, downsample)

        if c_in != c_out or downsample:
            self.skip = conv1x1(c_in, c_out, downsample)
        else:
            self.skip = Identity()

    def forward(self, x):
        return self.conv(x) + self.skip(x)


def BasicNetwork(c_in, num_classes):
    return nn.Sequential(
            conv3x3(c_in, 64),

            ResNetBlock(64, 64),
            ResNetBlock(64, 64),

            ResNetBlock(64, 128, True),
            ResNetBlock(128, 128),

            ResNetBlock(128, 256, True),
            ResNetBlock(256, 256),

            ResNetBlock(256, 512, True),
            ResNetBlock(512, 512),

            nn.AdaptiveAvgPool2d(1),
            Squeeze(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes))
