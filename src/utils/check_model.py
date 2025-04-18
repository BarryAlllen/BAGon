
import os
import torch

from torchinfo import summary

from src.networks.discriminator import Discriminator as Discriminator
from src.networks.model import BAGon
from src.networks.noise_layers import ScreenShootingNoiseLayer

# show the info of model
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


encoder_input_size = [(1, 3, 128, 128), (1, 15)]
encoder_input_size1 = [(1, 3, 128, 128), (1, 30)]
encoder_input_size2 = [(1, 3, 256, 256), (1, 30)]
decoder_input_size1 = [1, 3, 128, 128]
decoder_input_size2 = [1, 3, 256, 256]
discriminator_input_size = [1, 3, 128, 128]
noise_input_size = [1, 3, 128, 128]


def encoder_info(input_size):
    model = BAGon().encoder.to(device)
    print(f"{model.name}\n{model}")
    summary(model, input_size=input_size)


def decoder_info(input_size):
    model = BAGon().decoder.to(device)
    print(f"{model.name}\n{model}")
    summary(model, input_size=input_size)


def disciminator_info(input_size):
    model = Discriminator(in_channels=3, hidden_channels=64).to(device)
    print(f"{model.name}\n{model}")
    summary(model, input_size=input_size)


def noise_info(input_size):
    model = ScreenShootingNoiseLayer().to(device)
    print(f"{model.name}\n{model}")
    summary(model, input_size=input_size)


def main():
    # encoder_info(encoder_input_size1)
    decoder_info(decoder_input_size1)
    # disciminator_info(discriminator_input_size)
    # noise_info(noise_input_size)


if __name__ == '__main__':
    main()


