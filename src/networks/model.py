
import torch.nn as nn

from src.networks.decoder import Decoder
from src.networks.encoder import Encoder


class BAGonEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, out_channels=out_channels)

    def forward(self, image, message):
        image_encoded = self.encoder(image, message)
        return image_encoded


class BAGonDecoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.decoder = Decoder(in_channels=in_channels)

    def forward(self, image_encoded):
        message_decoded = self.decoder(image_encoded)
        return message_decoded