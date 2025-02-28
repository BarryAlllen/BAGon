
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
        print("Model: Decoder")
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


class ExtractorV2(nn.Module):
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
        self.layer5 = nn.Conv2d(64, 64, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(64, 30),
            nn.Sigmoid()
        )
        self.attention_1 = ConvolutionalBlockAttentionModule(64)
        # self.attention_2 = ConvolutionalBlockAttentionModule(64)
        # self.attention_3 = ConvolutionalBlockAttentionModule(64)
        # self.attention_4 = ConvolutionalBlockAttentionModule(64)

    def forward(self, x):
        out = self.layer1(x)
        out = self.attention_1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class DecoderV2(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64):
        super().__init__()
        print("Model: DecoderV2")
        self.layer1 = nn.Sequential(
            ConvBatchNormReluBlock(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1),
            ConvBatchNormReluBlock(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1),
            ConvBatchNormReluBlock(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1),
            ResidualBlock(in_channels=hidden_channels, out_channels=hidden_channels, stride=1),
            ResidualBlock(in_channels=hidden_channels, out_channels=hidden_channels, stride=1),
            ResidualBlock(in_channels=hidden_channels, out_channels=hidden_channels, stride=1),
        )
        self.extractor = ExtractorV2(in_channels=hidden_channels)

    def forward(self, x):
        out = self.layer1(x)
        out = self.extractor(out)
        return out
