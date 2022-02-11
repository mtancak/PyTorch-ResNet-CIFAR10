import torch
import torch.nn as nn


def conv0(in_features, out_features = 64):
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


def conv_x(in_features, mid_features, out_features):
    return nn.ModuleList([
            nn.Conv2d(
                in_channels=in_features,
                out_channels=mid_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_features,
                  out_channels=mid_features,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_features,
                  out_channels=out_features,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=False),
            nn.BatchNorm2d(out_features)
        ])


def skip_x(in_features, out_features):
    return nn.ModuleList([
            nn.Conv2d(in_channels=in_features,
                  out_channels=out_features,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=False),
            nn.BatchNorm2d(out_features)
        ])


class ResNet50(nn.Module):
    def __init__(self, num_channels=3):
        super(ResNet50, self).__init__()

        self.blocks = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        block_repeats_lis = [3, 4, 6, 3]
        in_features_lis = [64, 256, 512, 1024]
        mid_features_lis = [64, 128, 256, 512]
        scale = 4

        self.conv0_block = conv0(num_channels)

        for repeat, in_features, mid_features in zip(block_repeats_lis, in_features_lis, mid_features_lis):
            self.blocks.append(conv_x(in_features, mid_features, scale*mid_features))
            self.skip_layers.append(skip_x(in_features, scale*mid_features))
            for i in range(repeat):
                self.blocks.append(conv_x(scale*mid_features, mid_features, scale*mid_features))
                self.skip_layers.append(nn.ModuleList([nn.Identity()]))

    def forward(self, x):
        x_ = x

        for module in self.conv0_block:
            x_ = module(x_)

        print("num layers = " + str(len(self.blocks)))
        for block, skip_layer in zip(self.blocks, self.skip_layers):
            print("layer, x_.shape = " + str(x_.shape))

            interm_x_ = x_.clone()

            for module in block:
                print("module")
                interm_x_ = module(interm_x_)
                print(" shape = " + str(interm_x_.shape))

            skip_x_ = x_.clone()

            for skip_module in skip_layer:
                skip_x_ = skip_module(skip_x_)

            x_ = interm_x_ + skip_x_

        return x_


if __name__ == "__main__":
    model = ResNet50(num_channels=3)
    model(torch.rand((1, 3, 224, 224)))
