
import os
import torch

from torch import nn
from src.networks.discriminator import Discriminator
from src.networks.model import BAGon

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_model(weights_path: str, device):
    model = BAGon().to(device)
    model = load_weights(model, weights_path, device)
    return model


def get_discriminator(weights_path: str, device):
    discriminator = Discriminator().to(device)
    discriminator = load_weights(discriminator, weights_path, device)
    return discriminator


def load_weights(model: nn.Module, weights_path: str, device):
    # load weights
    checkpoint = torch.load(weights_path, map_location=device)
    weights_dict = {}
    for k, v in checkpoint.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        if v.device.type != device.type:  # Check if weights
            raise RuntimeError(f"Weight {new_k} is not on device {device}, but on {new_k.device} instead.")
        weights_dict[new_k] = v
    model.load_state_dict(weights_dict)
    return model


# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = get_model(
#         weights_path='/workspace/code/watermark/bagon/data/result/25-03-03_18h-03m-12s/model/model.pth',
#         device=device
#     )
#     print(type(model))
#     print(next(model.parameters()).device)
#
#
# if __name__ == '__main__':
#     main()