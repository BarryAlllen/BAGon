import torch
import torch.nn as nn

from src.networks.common import DoubleConvBatchNormReluBlock, DownsampleMaxPoolBlock, UpsampleConvBlock


class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.conv_block_1 = DoubleConvBatchNormReluBlock(in_channels=in_channels, out_channels=16)
        self.conv_block_2 = DoubleConvBatchNormReluBlock(in_channels=16, out_channels=32)
        self.conv_block_3 = DoubleConvBatchNormReluBlock(in_channels=32, out_channels=64)
        self.conv_block_4 = DoubleConvBatchNormReluBlock(in_channels=64 * 3, out_channels=64)
        self.conv_block_5 = DoubleConvBatchNormReluBlock(in_channels=32 * 2 + 64, out_channels=32)
        self.conv_block_6 = DoubleConvBatchNormReluBlock(in_channels=16 * 2 + 64, out_channels=16)

        self.down_sample_2x2 = DownsampleMaxPoolBlock(kernel_size=2, stride=2)
        self.down_sample_4x4 = DownsampleMaxPoolBlock(kernel_size=4, stride=4)

        self.up_sample_1 = UpsampleConvBlock(in_channels=64 * 3, out_channels=64)
        self.up_sample_2 = UpsampleConvBlock(in_channels=64, out_channels=32)
        self.up_sample_3 = UpsampleConvBlock(in_channels=32, out_channels=16)

        self.linear_message = nn.Linear(in_features=30, out_features=256)
        self.conv_message = DoubleConvBatchNormReluBlock(in_channels=1, out_channels=64)

        self.channel_adjust = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.interpolate = nn.functional.interpolate

    def message_process(self, message, height, width, view_size, is_interpolate=True, mode='bilinear'):
        message = self.linear_message(message)
        message = message.view(-1, 1, view_size, view_size)
        if is_interpolate:
            message = self.interpolate(message, size=(height, width), mode=mode)
        message = self.conv_message(message)
        return message

    def forward(self, x, message):
        x1 = self.conv_block_1(x)

        x2 = self.down_sample_2x2(x1)
        x2 = self.conv_block_2(x2)

        x3 = self.down_sample_2x2(x2)
        x3 = self.conv_block_3(x3)

        x4 = self.down_sample_2x2(x3)

        x5 = self.down_sample_4x4(x4)
        x6 = x5.repeat(1, 1, 4, 4)

        return x
