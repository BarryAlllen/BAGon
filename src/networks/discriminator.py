
import torch.nn as nn

from src.networks.attention import ConvolutionalBlockAttentionModule
from src.networks.common import ConvBatchNormReluBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64):
        super().__init__()
        self.name = "Model: Discriminator"
        self.layers = nn.Sequential(
            ConvBatchNormReluBlock(in_channels=in_channels, out_channels=hidden_channels),
            ConvBatchNormReluBlock(in_channels=hidden_channels, out_channels=hidden_channels),
            ConvBatchNormReluBlock(in_channels=hidden_channels, out_channels=hidden_channels),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        x = self.layers(x)
        x.squeeze_(3).squeeze_(2)
        out = self.linear(x)
        return out


class DiscriminatorV2(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64):
        super().__init__()
        self.name = "Model: DiscriminatorV2"
        self.layers = nn.Sequential(
            ConvBatchNormReluBlock(in_channels=in_channels, out_channels=hidden_channels),
            ConvolutionalBlockAttentionModule(in_channels=hidden_channels),
            ConvBatchNormReluBlock(in_channels=hidden_channels, out_channels=hidden_channels),
            ConvolutionalBlockAttentionModule(in_channels=hidden_channels),
            ConvBatchNormReluBlock(in_channels=hidden_channels, out_channels=hidden_channels),
            ConvolutionalBlockAttentionModule(in_channels=hidden_channels),
            ConvBatchNormReluBlock(in_channels=hidden_channels, out_channels=hidden_channels),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        x = self.layers(x)
        x.squeeze_(3).squeeze_(2)
        out = self.linear(x)
        return out