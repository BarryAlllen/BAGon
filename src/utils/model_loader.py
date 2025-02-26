import torch

from src.networks.model import BAGon


def get_model(params_path, device):
    model = BAGon().to(device)

    checkpoint = torch.load(params_path)

    weights_dict = {}
    for k, v in checkpoint.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    model.load_state_dict(weights_dict)
    return model