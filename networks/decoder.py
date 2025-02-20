import torch.nn as nn

from networks.attention import ConvolutionalBlockAttentionModule
from networks.common import ConvBatchNormReluBlock, ResidualBlock


class Decoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # self.attention = ConvolutionalBlockAttentionModule(64)
        #
        # self.conv_block_1 = ConvBatchNormReluBlock(in_channels=64, out_channels=64)
        # self.conv_block_2 = ConvBatchNormReluBlock(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        #
        # self.residual_block_1 = nn.Sequential(
        #     ConvBatchNormReluBlock(in_channels=in_channels, out_channels=64),
        #     ConvBatchNormReluBlock(in_channels=64, out_channels=64),
        #     ConvBatchNormReluBlock(in_channels=64, out_channels=64),
        #     ResidualBlock(in_channels=64, out_channels=64, stride=1),
        #     ResidualBlock(in_channels=64, out_channels=64, stride=1),
        #     ResidualBlock(in_channels=64, out_channels=64, stride=1)
        # )
        #
        # self.residual_block_2 = nn.Sequential(
        #     ResidualBlock(in_channels=64, out_channels=64, stride=1),
        #     ResidualBlock(in_channels=64, out_channels=64, stride=2)
        # )
        #
        # self.residual_block_3 = nn.Sequential(
        #     ResidualBlock(in_channels=64, out_channels=64, stride=1),
        #     ResidualBlock(in_channels=64, out_channels=64, stride=2)
        # )
        #
        # self.residual_block_4 = nn.Sequential(
        #     ResidualBlock(in_channels=64, out_channels=64, stride=1),
        #     ResidualBlock(in_channels=64, out_channels=64, stride=2)
        # )

        self.layers = nn.Sequential(
            ConvBatchNormReluBlock(in_channels=in_channels, out_channels=64),
            ConvBatchNormReluBlock(in_channels=64, out_channels=64),
            ConvBatchNormReluBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64, stride=1),
            ResidualBlock(in_channels=64, out_channels=64, stride=1),
            ResidualBlock(in_channels=64, out_channels=64, stride=1),
            ConvBatchNormReluBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64, stride=1),
            ResidualBlock(in_channels=64, out_channels=64, stride=2),
            ResidualBlock(in_channels=64, out_channels=64, stride=1),
            ResidualBlock(in_channels=64, out_channels=64, stride=2),
            ResidualBlock(in_channels=64, out_channels=64, stride=1),
            ResidualBlock(in_channels=64, out_channels=64, stride=2),
            ConvolutionalBlockAttentionModule(64),
            ConvBatchNormReluBlock(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.linear = nn.Linear(in_features=256, out_features=30)

    def forward(self, x):
        x = self.layers(x)
        x.squeeze_(1)
        x = x.view(-1, 1, 256)
        out = self.linear(x)
        out.squeeze_(1)
        return out

