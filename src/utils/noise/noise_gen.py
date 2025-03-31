import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch

from PIL import Image
from torchvision import datasets, transforms
from src.utils.noise.noise_layers import JPEGNoise, PerspectiveNoise, LightNoise, MoireNoise, GaussianNoise, ScreenShootingNoiseLayer

device = torch.device("cuda")

image = Image.open("/workspace/code/watermark/temp_image/12.jpg")

image = transforms.ToTensor()(image)

image = image.unsqueeze(0).to(device)
print("Image Shape:", image.shape)

noise_layer = ScreenShootingNoiseLayer()
noised_image = noise_layer(image)
print("Noised Image Shape:", noised_image.shape)

def save_tensor_as_image(tensor, filename):
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.cpu().detach()
    tensor = tensor.clamp(0, 1)
    pil_image = transforms.ToPILImage()(tensor)
    pil_image.save(filename)


noise_mapping = {
    "jpg": JPEGNoise(),
    "perspective": PerspectiveNoise(),
    "light": LightNoise(),
    "moire": MoireNoise(),
    "gaussian": GaussianNoise()
}

save_tensor_as_image(noised_image, "/workspace/code/watermark/temp_noise/noised_image.jpg")