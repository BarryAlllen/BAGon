
import torch
import torch.nn as nn

from src.networks.attention import ConvolutionalBlockAttentionModule
from src.networks.common import DoubleConvBatchNormReluBlock, DownsampleMaxPoolBlock, UpsampleConvBlock, \
    ConvBatchNormReluBlock


class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.name = f"Model: {self.__class__.__name__}"

        self.conv_block_1 = DoubleConvBatchNormReluBlock(in_channels=in_channels, out_channels=16)
        self.conv_block_2 = DoubleConvBatchNormReluBlock(in_channels=16, out_channels=32)
        self.conv_block_3 = DoubleConvBatchNormReluBlock(in_channels=32, out_channels=64)

        self.down_sample_2x2 = DownsampleMaxPoolBlock(kernel_size=2, stride=2)
        self.down_sample_4x4 = DownsampleMaxPoolBlock(kernel_size=4, stride=4)

        self.up_sample_1 = nn.Upsample(scale_factor=2)
        self.up_sample_2 = nn.Upsample(scale_factor=2)
        self.up_sample_3 = nn.Upsample(scale_factor=2)
        self.up_sample_4 = nn.Upsample(scale_factor=2)

        self.conv_block_1_1 = ConvBatchNormReluBlock(in_channels=16 * 2 + 64, out_channels=16)
        self.conv_block_2_1 = ConvBatchNormReluBlock(in_channels=32 * 2 + 64, out_channels=32)
        self.conv_block_2_2 = ConvBatchNormReluBlock(in_channels=32, out_channels=16)
        self.conv_block_3_1 = ConvBatchNormReluBlock(in_channels=64 * 2 + 64, out_channels=64)
        self.conv_block_3_2 = ConvBatchNormReluBlock(in_channels=64, out_channels=32)
        self.conv_block_4_1 = ConvBatchNormReluBlock(in_channels=64 * 2 + 64, out_channels=64)
        self.conv_block_4_2 = ConvBatchNormReluBlock(in_channels=64, out_channels=64)

        self.attention_1 = ConvolutionalBlockAttentionModule(16)
        self.attention_2 = ConvolutionalBlockAttentionModule(32)
        self.attention_3 = ConvolutionalBlockAttentionModule(64)
        self.attention_4 = ConvolutionalBlockAttentionModule(64)

        self.linear_message = nn.Linear(in_features=30, out_features=256)
        self.conv_message = DoubleConvBatchNormReluBlock(in_channels=1, out_channels=64)

        self.channel_adjust = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.interpolate = nn.functional.interpolate

    def message_process(self, message, view_size=16, is_interpolate=True, height=None, width=None, mode='bilinear'):
        message = self.linear_message(message)
        message = message.view(-1, 1, view_size, view_size)
        if is_interpolate:
            assert height is not None and width is not None, "Wrong input for height and width"
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

        x44 = x5.repeat(1, 1, 4, 4)
        message_4 = self.message_process(message=message, is_interpolate=False)
        x44 = torch.cat(tensors=(x4, x44, message_4), dim=1)
        x44 = self.conv_block_4_1(x44)
        x44 = self.attention_4(x44)
        x44 = self.conv_block_4_2(x44)

        x33 = self.up_sample_3(x44)
        message_3 = self.message_process(message=message, height=x33.shape[2], width=x33.shape[3])
        x33 = torch.cat(tensors=(x3, x33, message_3), dim=1)
        x33 = self.conv_block_3_1(x33)
        x33 = self.attention_3(x33)
        x33 = self.conv_block_3_2(x33)

        x22 = self.up_sample_2(x33)
        message_2 = self.message_process(message=message, height=x22.shape[2], width=x22.shape[3])
        x22 = torch.cat(tensors=(x2, x22, message_2), dim=1)
        x22 = self.conv_block_2_1(x22)
        x22 = self.attention_2(x22)
        x22 = self.conv_block_2_2(x22)

        x11 = self.up_sample_1(x22)
        message_1 = self.message_process(message=message, height=x11.shape[2], width=x11.shape[3])
        x11 = torch.cat(tensors=(x1, x11, message_1), dim=1)
        x11 = self.conv_block_1_1(x11)
        x11 = self.attention_1(x11)

        out = self.channel_adjust(x11)
        return out


class EncoderV1(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.name = f"Model: {self.__class__.__name__}"

        self.conv_block_1 = DoubleConvBatchNormReluBlock(in_channels=in_channels, out_channels=16)
        self.conv_block_2 = DoubleConvBatchNormReluBlock(in_channels=16, out_channels=32)
        self.conv_block_3 = DoubleConvBatchNormReluBlock(in_channels=32, out_channels=64)

        self.conv_block_11 = DoubleConvBatchNormReluBlock(in_channels=16 * 2 + 64, out_channels=16)
        self.conv_block_22 = DoubleConvBatchNormReluBlock(in_channels=32 * 2 + 64, out_channels=32)
        self.conv_block_33 = DoubleConvBatchNormReluBlock(in_channels=64 * 3, out_channels=64)

        self.down_sample_2x2 = DownsampleMaxPoolBlock(kernel_size=2, stride=2)
        self.down_sample_4x4 = DownsampleMaxPoolBlock(kernel_size=4, stride=4)

        self.up_sample_1 = UpsampleConvBlock(in_channels=32, out_channels=16)
        self.up_sample_2 = UpsampleConvBlock(in_channels=64, out_channels=32)
        self.up_sample_3 = UpsampleConvBlock(in_channels=64 * 3, out_channels=64)

        self.attention_1 = ConvolutionalBlockAttentionModule(16)
        self.attention_2 = ConvolutionalBlockAttentionModule(32)
        self.attention_3 = ConvolutionalBlockAttentionModule(64)

        self.linear_message = nn.Linear(in_features=30, out_features=256)
        self.conv_message = DoubleConvBatchNormReluBlock(in_channels=1, out_channels=64)

        self.channel_adjust = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.interpolate = nn.functional.interpolate

    def message_process(self, message, view_size=16, is_interpolate=True, height=None, width=None, mode='bilinear'):
        message = self.linear_message(message)
        message = message.view(-1, 1, view_size, view_size)
        if is_interpolate:
            assert height is not None and width is not None, "Wrong input for height and width"
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

        x44 = x5.repeat(1, 1, 4, 4)
        message_4 = self.message_process(message=message,  is_interpolate=False)
        x4 = torch.cat(tensors=(x4, x44, message_4), dim=1)

        x33 = self.up_sample_3(x4)
        x33 = self.attention_3(x33)
        message_3 = self.message_process(message=message, height=x33.shape[2], width=x33.shape[3])
        x33 = torch.cat(tensors=(x3, x33, message_3), dim=1)
        x33 = self.conv_block_33(x33)

        x22 = self.up_sample_2(x33)
        x22 = self.attention_2(x22)
        message_2 = self.message_process(message=message, height=x22.shape[2], width=x22.shape[3])
        x22 = torch.cat(tensors=(x2, x22, message_2), dim=1)
        x22 = self.conv_block_22(x22)

        x11 = self.up_sample_1(x22)
        x11 = self.attention_1(x11)
        message_1 = self.message_process(message=message, height=x11.shape[2], width=x11.shape[3])
        x11 = torch.cat(tensors=(x1, x11, message_1), dim=1)
        x11 = self.conv_block_11(x11)

        out = self.channel_adjust(x11)
        return out


class EncoderV2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.name = f"Model: {self.__class__.__name__}"

        # self.conv_block_1 = DoubleConvBatchNormReluBlock(in_channels=in_channels, out_channels=8)
        self.conv_block_1 = DoubleConvBatchNormReluBlock(in_channels=in_channels, out_channels=16)
        self.conv_block_2 = DoubleConvBatchNormReluBlock(in_channels=16, out_channels=32)
        self.conv_block_3 = DoubleConvBatchNormReluBlock(in_channels=32, out_channels=64)
        self.conv_block_4 = DoubleConvBatchNormReluBlock(in_channels=64, out_channels=128)

        self.down_sample_2x2 = DownsampleMaxPoolBlock(kernel_size=2, stride=2)
        self.down_sample_4x4 = DownsampleMaxPoolBlock(kernel_size=4, stride=4)

        self.up_sample_1 = nn.Upsample(scale_factor=2)
        self.up_sample_2 = nn.Upsample(scale_factor=2)
        self.up_sample_3 = nn.Upsample(scale_factor=2)
        self.up_sample_4 = nn.Upsample(scale_factor=2)

        self.conv_block_11 = ConvBatchNormReluBlock(in_channels=16 * 2 + 64, out_channels=16)
        # self.conv_block_111 = ConvBatchNormReluBlock(in_channels=16, out_channels=8)
        self.conv_block_22 = ConvBatchNormReluBlock(in_channels=32 * 2 + 64, out_channels=32)
        self.conv_block_222 = ConvBatchNormReluBlock(in_channels=32, out_channels=16)
        self.conv_block_33 = ConvBatchNormReluBlock(in_channels=64 * 2 + 64, out_channels=64)
        self.conv_block_333 = ConvBatchNormReluBlock(in_channels=64, out_channels=32)
        self.conv_block_44 = ConvBatchNormReluBlock(in_channels=128 * 2 + 64, out_channels=128)
        self.conv_block_444 = ConvBatchNormReluBlock(in_channels=128, out_channels=64)

        self.attention_1 = ConvolutionalBlockAttentionModule(16)
        self.attention_2 = ConvolutionalBlockAttentionModule(32)
        self.attention_3 = ConvolutionalBlockAttentionModule(64)
        self.attention_4 = ConvolutionalBlockAttentionModule(128)

        self.linear_message = nn.Linear(in_features=30, out_features=256)
        self.conv_message = DoubleConvBatchNormReluBlock(in_channels=1, out_channels=64)

        self.channel_adjust = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.interpolate = nn.functional.interpolate

    def message_process(self, message, view_size=16, is_interpolate=True, height=None, width=None, mode='bilinear'):
        message = self.linear_message(message)
        message = message.view(-1, 1, view_size, view_size)
        if is_interpolate:
            assert height is not None and width is not None, "Wrong input for height and width"
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
        x4 = self.conv_block_4(x4)

        x5 = self.down_sample_4x4(x4)

        x44 = x5.repeat(1, 1, 4, 4)
        message_4 = self.message_process(message=message, is_interpolate=False)
        x44 = torch.cat(tensors=(x4, x44, message_4), dim=1)
        x44 = self.conv_block_44(x44)
        x44 = self.attention_4(x44)
        x44 = self.conv_block_444(x44)

        x33 = self.up_sample_3(x44)
        message_3 = self.message_process(message=message, height=x33.shape[2], width=x33.shape[3])
        x33 = torch.cat(tensors=(x3, x33, message_3), dim=1)
        x33 = self.conv_block_33(x33)
        x33 = self.attention_3(x33)
        x33 = self.conv_block_333(x33)

        x22 = self.up_sample_2(x33)
        message_2 = self.message_process(message=message, height=x22.shape[2], width=x22.shape[3])
        x22 = torch.cat(tensors=(x2, x22, message_2), dim=1)
        x22 = self.conv_block_22(x22)
        x22 = self.attention_2(x22)
        x22 = self.conv_block_222(x22)

        x11 = self.up_sample_1(x22)
        message_1 = self.message_process(message=message, height=x11.shape[2], width=x11.shape[3])
        x11 = torch.cat(tensors=(x1, x11, message_1), dim=1)
        x11 = self.conv_block_11(x11)
        x11 = self.attention_1(x11)
        # x11 = self.conv_block_111(x11)

        out = self.channel_adjust(x11)
        return out


class EncoderV3(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.name = f"Model: {self.__class__.__name__}"

        self.conv_block_1 = DoubleConvBatchNormReluBlock(in_channels=in_channels, out_channels=16)
        self.conv_block_2 = DoubleConvBatchNormReluBlock(in_channels=16, out_channels=32)
        self.conv_block_3 = DoubleConvBatchNormReluBlock(in_channels=32, out_channels=64)

        self.down_sample_2x2 = DownsampleMaxPoolBlock(kernel_size=2, stride=2)
        self.down_sample_4x4 = DownsampleMaxPoolBlock(kernel_size=4, stride=4)

        self.up_sample_1 = nn.Upsample(scale_factor=2)
        self.up_sample_2 = nn.Upsample(scale_factor=2)
        self.up_sample_3 = nn.Upsample(scale_factor=2)
        self.up_sample_4 = nn.Upsample(scale_factor=2)

        self.conv_block_11 = ConvBatchNormReluBlock(in_channels=16 * 2 + 64, out_channels=16)
        self.conv_block_22 = ConvBatchNormReluBlock(in_channels=32 * 2 + 64, out_channels=32)
        self.conv_block_222 = ConvBatchNormReluBlock(in_channels=32, out_channels=16)
        self.conv_block_33 = ConvBatchNormReluBlock(in_channels=64 * 2 + 64, out_channels=64)
        self.conv_block_333 = ConvBatchNormReluBlock(in_channels=64, out_channels=32)
        self.conv_block_44 = ConvBatchNormReluBlock(in_channels=64 * 2 + 64, out_channels=64)
        self.conv_block_444 = ConvBatchNormReluBlock(in_channels=64, out_channels=64)

        self.attention_1 = ConvolutionalBlockAttentionModule(16)
        self.attention_2 = ConvolutionalBlockAttentionModule(32)
        self.attention_3 = ConvolutionalBlockAttentionModule(64)
        self.attention_4 = ConvolutionalBlockAttentionModule(64)

        self.linear_message = nn.Linear(in_features=30, out_features=256)
        self.conv_message = DoubleConvBatchNormReluBlock(in_channels=1, out_channels=64)

        self.channel_adjust = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.interpolate = nn.functional.interpolate

    def message_process(self, message, view_size=16, is_interpolate=True, height=None, width=None, mode='bilinear'):
        message = self.linear_message(message)
        message = message.view(-1, 1, view_size, view_size)
        if is_interpolate:
            assert height is not None and width is not None, "Wrong input for height and width"
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

        x44 = x5.repeat(1, 1, 4, 4)
        message_4 = self.message_process(message=message, is_interpolate=False)
        x44 = torch.cat(tensors=(x4, x44, message_4), dim=1)
        x44 = self.conv_block_44(x44)
        x44 = self.attention_4(x44)
        x44 = self.conv_block_444(x44)

        x33 = self.up_sample_3(x44)
        message_3 = self.message_process(message=message, height=x33.shape[2], width=x33.shape[3])
        x33 = torch.cat(tensors=(x3, x33, message_3), dim=1)
        x33 = self.conv_block_33(x33)
        x33 = self.attention_3(x33)
        x33 = self.conv_block_333(x33)

        x22 = self.up_sample_2(x33)
        message_2 = self.message_process(message=message, height=x22.shape[2], width=x22.shape[3])
        x22 = torch.cat(tensors=(x2, x22, message_2), dim=1)
        x22 = self.conv_block_22(x22)
        x22 = self.attention_2(x22)
        x22 = self.conv_block_222(x22)

        x11 = self.up_sample_1(x22)
        message_1 = self.message_process(message=message, height=x11.shape[2], width=x11.shape[3])
        x11 = torch.cat(tensors=(x1, x11, message_1), dim=1)
        x11 = self.conv_block_11(x11)
        x11 = self.attention_1(x11)

        out = self.channel_adjust(x11)
        return out