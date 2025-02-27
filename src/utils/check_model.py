
from torchinfo import summary

import torch

from src.networks.decoder import Decoder
from src.networks.discriminator import Discriminator
from src.networks.encoder import EncoderV2
from src.networks.noise_layers import ScreenShootingNoiseLayer

# show the info of model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

encoder_input_size = [(1, 3, 128, 128), (1, 15)]
encoder_input_size1 = [(1, 3, 128, 128), (1, 30)]
encoder_input_size2 = [(1, 3, 256, 256), (1, 30)]
decoder_input_size1 = [1, 3, 128, 128]
decoder_input_size2 = [1, 3, 256, 256]
discriminator_input_size = [1, 3, 1, 2]
noise_input_size = [3, 1, 1, 2]

def encoder_info(input_size):
    model = EncoderV2().to(device)
    print(model)
    summary(model, input_size=input_size)

def decoder_info(input_size):
    model = Decoder().to(device)
    print(model)
    summary(model, input_size=input_size)

def disciminator_info(input_size):
    model = Discriminator(64).to(device)
    print(model)
    summary(model, input_size=input_size)

def noise_info(input_size):
    model = ScreenShootingNoiseLayer().to(device)
    print(model)
    summary(model, input_size=input_size)

def main():
    encoder_info(encoder_input_size1)
    # decoder_info(decoder_input_size1)
    # disciminator_info(discriminator_input_size)
    # noise_info(noise_input_size)

if __name__ == '__main__':
    main()


