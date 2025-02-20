import torch
import torch.nn as nn

from src.networks.decoder import Decoder
from src.networks.encoder import Encoder
from src.networks.noise_layers import ScreenShootingNoiseLayer, NoneNoiseLayer

torch.autograd.set_detect_anomaly(True)
class BAGon(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, is_noise=True):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, out_channels=out_channels)
        self.decoder = Decoder(in_channels=in_channels)
        if is_noise:
            self.noise_layer = ScreenShootingNoiseLayer()
        else:
            self.noise_layer = NoneNoiseLayer()

    def forward(self, image, message):
        image_encoded = self.encoder(image, message)
        image_encoded_noised = self.noise_layer(image_encoded)
        message_decoded = self.decoder(image_encoded_noised)
        return image_encoded, image_encoded_noised, message_decoded