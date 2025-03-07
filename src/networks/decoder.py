
import torch.nn as nn

from src.networks.attention import ConvolutionalBlockAttentionModule
from src.networks.common import ConvBatchNormReluBlock, ResidualBlock


class Extractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer1 = ConvBatchNormReluBlock(in_channels, 64, 3, 1)
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 2)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 2)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 2)
        )
        self.layer5 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear = nn.Linear(256, 30)
        self.attention = ConvolutionalBlockAttentionModule(64)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.attention(out)
        out = self.layer5(out)
        out.squeeze_(1)
        out = out.view(-1, 1, 256)
        out = self.linear(out)
        out.squeeze_(1)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.name = f"Model: {self.__class__.__name__}"

        self.layer1 = nn.Sequential(
            ConvBatchNormReluBlock(in_channels, 64, 3, 1),
            ConvBatchNormReluBlock(64, 64, 3, 1),
            ConvBatchNormReluBlock(64, 64, 3, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
        )
        self.extractor = Extractor(in_channels=64)

    def forward(self, x):
        out = self.layer1(x)
        out = self.extractor(out)
        return out


class ExtractorV0(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer1 = ConvBatchNormReluBlock(in_channels, 64, 3, 1)
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 2)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 2)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 2)
        )
        self.layer5 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear = nn.Linear(256, 30)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out.squeeze_(1)
        out = out.view(-1, 1, 256)
        out = self.linear(out)
        out.squeeze_(1)
        return out


class DecoderV0(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.name = f"Model: {self.__class__.__name__}"

        self.layer1 = nn.Sequential(
            ConvBatchNormReluBlock(in_channels, 64, 3, 1),
            ConvBatchNormReluBlock(64, 64, 3, 1),
            ConvBatchNormReluBlock(64, 64, 3, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
        )
        self.extractor = Extractor(in_channels=64)

    def forward(self, x):
        out = self.layer1(x)
        out = self.extractor(out)
        return out


class ExtractorV3(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.layer1 = ConvBatchNormReluBlock(in_channels, hidden_channels, 3, 1)
        self.layer2 = nn.Sequential(
            ResidualBlock(hidden_channels, hidden_channels, 1),
            ResidualBlock(hidden_channels, hidden_channels, 2)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(hidden_channels, hidden_channels, 1),
            ResidualBlock(hidden_channels, hidden_channels, 2)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(hidden_channels, hidden_channels, 1),
            ResidualBlock(hidden_channels, hidden_channels, 2)
        )
        self.layer5 = nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear = nn.Linear(256, 30)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out.squeeze_(1)
        out = out.view(-1, 1, 256)
        out = self.linear(out)
        out.squeeze_(1)
        return out


class DecoderV3(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128):
        super().__init__()
        self.name = f"Model: {self.__class__.__name__}"

        self.layer1 = nn.Sequential(
            ConvBatchNormReluBlock(in_channels, hidden_channels, 3, 1),
            ConvBatchNormReluBlock(hidden_channels, hidden_channels, 3, 1),
            ConvBatchNormReluBlock(hidden_channels, hidden_channels, 3, 1),
            ResidualBlock(hidden_channels, hidden_channels, 1),
            ResidualBlock(hidden_channels, hidden_channels, 1),
            ResidualBlock(hidden_channels, hidden_channels, 1),
        )
        self.extractor = Extractor(in_channels=hidden_channels)

    def forward(self, x):
        out = self.layer1(x)
        out = self.extractor(out)
        return out