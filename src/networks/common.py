import torch.nn as nn

class ConvBatchNormReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class DoubleConvBatchNormReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBatchNormReluBlock(in_channels, out_channels, kernel_size, stride, padding),
            ConvBatchNormReluBlock(out_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self, x):
        return self.layers(x)


class MultiConvBatchNormReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, kernel_size=3, stride=1, padding=1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(ConvBatchNormReluBlock(in_channels, out_channels, kernel_size, stride, padding))
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.relu = nn.ReLU()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layers(x)
        out += self.shortcut(x)
        return self.relu(out)


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBatchNormReluBlock(in_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self, x):
        return self.layers(x)


class DownsampleMaxPoolBlock(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        )

    def forward(self, x):
        return self.layers(x)