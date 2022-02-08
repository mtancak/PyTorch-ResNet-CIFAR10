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
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    ])


def conv1_x(first_layer=True):
    interm_features = 64

    if first_layer:
        skip_module = nn.ModuleList([
            nn.Conv2d(in_channels=interm_features,
                out_channels=4*interm_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(4*interm_features)])
    else:
        skip_module = nn.ModuleList([nn.Identity()])

    return nn.ModuleList([
            nn.Conv2d(
                in_channels=interm_features if first_layer else 4*interm_features,
                out_channels=interm_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(interm_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=interm_features,
                  out_channels=interm_features,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
            nn.BatchNorm2d(interm_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=interm_features,
                  out_channels=4*interm_features,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=False),
            nn.BatchNorm2d(4*interm_features)
        ]), skip_module


def conv2_x(first_layer=True):
    interm_features = 128

    if first_layer:
        skip_module = nn.ModuleList([
            nn.Conv2d(in_channels=interm_features,
                out_channels=4*interm_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(4*interm_features)])
    else:
        skip_module = nn.ModuleList([nn.Identity()])

    return nn.ModuleList([
            nn.Conv2d(
                in_channels=interm_features if first_layer else 4*interm_features,
                out_channels=interm_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(interm_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=interm_features,
                  out_channels=interm_features,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
            nn.BatchNorm2d(interm_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=interm_features,
                  out_channels=4*interm_features,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=False),
            nn.BatchNorm2d(4*interm_features)
        ]), skip_module


def conv3_x(first_layer=True):
    interm_features = 256

    if first_layer:
        skip_module = nn.ModuleList([
            nn.Conv2d(in_channels=interm_features,
                out_channels=4*interm_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(4*interm_features)])
    else:
        skip_module = nn.ModuleList([nn.Identity()])

    return nn.ModuleList([
            nn.Conv2d(
                in_channels=interm_features if first_layer else 4*interm_features,
                out_channels=interm_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(interm_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=interm_features,
                  out_channels=interm_features,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
            nn.BatchNorm2d(interm_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=interm_features,
                  out_channels=4*interm_features,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=False),
            nn.BatchNorm2d(4*interm_features)
        ]), skip_module


def conv4_x(first_layer=True):
    interm_features = 512

    if first_layer:
        skip_module = nn.ModuleList([
            nn.Conv2d(in_channels=interm_features,
                out_channels=4*interm_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(4*interm_features)])
    else:
        skip_module = nn.ModuleList([nn.Identity()])

    return nn.ModuleList([
            nn.Conv2d(
                in_channels=interm_features if first_layer else 4*interm_features,
                out_channels=interm_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(interm_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=interm_features,
                  out_channels=interm_features,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
            nn.BatchNorm2d(interm_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=interm_features,
                  out_channels=4*interm_features,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=False),
            nn.BatchNorm2d(4*interm_features)
        ]), skip_module


class ResNet50(nn.Module):
    def __init__(self, num_channels=3):
        super(ResNet50, self).__init__()

        self.layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        layer_funcs = [conv1_x, conv2_x, conv3_x, conv4_x]
        layer_repeats = [3, 4, 6, 3]

        self.layers.append(conv0(num_channels))
        self.skip_layers.append(None)

        for f, r in zip(layer_funcs, layer_repeats):
            print("create block")
            layer, skip_layer = f()
            self.layers.append(layer)
            self.skip_layers.append(skip_layer)
            for i in range(r-1):
                print("create block")
                interm_layer, interm_skip_layer = f(False)
                self.layers.append(interm_layer)
                self.skip_layers.append(interm_skip_layer)

    def forward(self, x):
        x_ = x

        print("num layers = " + str(len(self.layers)))
        for layer, skip_layer in zip(self.layers, self.skip_layers):
            print("layer, x_.shape = " + str(x_.shape))

            interm_x_ = x_.clone()
            for module in layer:
                print("module")
                interm_x_ = module(interm_x_)
                print(" shape = " + str(interm_x_.shape))

            x_ = interm_x_

            if skip_layer:
                skip_x_ = x_.clone()
                for skip_module in skip_layer:
                    skip_x_ = skip_module(skip_x_)
                x_ += skip_x_

        return x


if __name__ == "__main__":
    model = ResNet50(num_channels=3)
    model(torch.rand((1, 3, 224, 224)))
