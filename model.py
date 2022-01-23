import torch
import torch.nn as nn


def conv1(in_features):
    out_features = 64
    return nn.ModuleList([
        nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2)
    ])


def conv2_x():
    return nn.ModuleList([
        nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=1,
            bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64,
                  out_channels=64,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64,
                  out_channels=256,
                  kernel_size=1,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(256)
    ])


def conv3_x():
    return nn.ModuleList([
        nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=1,
            bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=128,
                  out_channels=128,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=128,
                  out_channels=512,
                  kernel_size=1,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(512)
    ])


def conv4_x():
    return nn.ModuleList([
        nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=1,
            bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256,
                  out_channels=256,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256,
                  out_channels=1024,
                  kernel_size=1,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(1024)
    ])


def conv5_x():
    return nn.ModuleList([
        nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=1,
            stride=1,
            padding=1,
            bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512,
                  out_channels=512,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512,
                  out_channels=2048,
                  kernel_size=1,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(2048)
    ])


class ResNet50(nn.Module):
    def __init__(self, num_channels=3):
        super(ResNet50, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(conv1(num_channels))
        self.layers.append(conv2_x())
        self.layers.append(conv3_x())
        self.layers.append(conv4_x())
        self.layers.append(conv5_x())

    def forward(self, x):
        for layer in self.layers:
            for module in layer:
                x = module(x)
                # skipped_connections.append(x)
        return x

    def display_attention(self, toggle):
        self.stack_of_encoders[-1].display_attention(toggle)


if __name__ == "__main__":
    c = 3
    model = ResNet50(num_channels=c)
    model(torch.rand((1, c, 32, 32)))
