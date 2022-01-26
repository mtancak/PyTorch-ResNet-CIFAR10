import torch
import torch.nn as nn


def conv0(in_features):
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
        nn.MaxPool2d(3, stride=2, padding=1)
    ])


def conv1_x(stride=1):
    out_features = 64
    return nn.ModuleList([
        nn.Conv2d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=1,
            stride=stride,
            padding=(0 if stride == 2 else 1),
            bias=False),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_features,
                  out_channels=out_features,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_features,
                  out_channels=4*out_features,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=False),
        nn.BatchNorm2d(4*out_features)
    ])


def conv2_x(stride=1):
    out_features = 128
    return nn.ModuleList([
        nn.Conv2d(
            in_channels=2*out_features,
            out_channels=out_features,
            kernel_size=1,
            stride=stride,
            padding=(0 if stride == 2 else 1),
            bias=False),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_features,
                  out_channels=out_features,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_features,
                  out_channels=4*out_features,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=False),
        nn.BatchNorm2d(4*out_features)
    ])


def conv3_x(stride=1):
    out_features = 256
    return nn.ModuleList([
        nn.Conv2d(
            in_channels=2*out_features,
            out_channels=out_features,
            kernel_size=1,
            stride=stride,
            padding=(0 if stride == 2 else 1),
            bias=False),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_features,
                  out_channels=out_features,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_features,
                  out_channels=4*out_features,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=False),
        nn.BatchNorm2d(4*out_features)
    ])


def conv4_x(stride=1):
    out_features = 512
    return nn.ModuleList([
        nn.Conv2d(
            in_channels=2*out_features,
            out_channels=out_features,
            kernel_size=1,
            stride=stride,
            padding=(0 if stride == 2 else 1),
            bias=False),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_features,
                  out_channels=out_features,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_features,
                  out_channels=4*out_features,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=False),
        nn.BatchNorm2d(4*out_features)
    ])


class ResNet50(nn.Module):
    def __init__(self, num_channels=3):
        super(ResNet50, self).__init__()

        self.layers = nn.ModuleList()
        layer_funcs = [conv1_x, conv2_x, conv3_x, conv4_x]
        layer_repeats = [3, 4, 6, 3]

        self.layers.append(conv0(num_channels))
        for f, r in zip(layer_funcs, layer_repeats):
            self.layers.append(f(2))
            for i in range(r-1):
                self.layers.append(f(1))

    def forward(self, x):
        x_ = x
        for layer in self.layers:
            print("layer")
            for module in layer:
                print("module")
                x_ = module(x_)
                print(" shape = " + str(x_.shape))
            # x = x_ + x
        return x

    def display_attention(self, toggle):
        self.stack_of_encoders[-1].display_attention(toggle)


if __name__ == "__main__":
    c = 3
    model = ResNet50(num_channels=c)
    model(torch.rand((1, c, 224, 224)))
