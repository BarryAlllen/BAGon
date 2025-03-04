import os

import numpy as np
import torch

from tqdm import tqdm

from src.networks.model import BAGon
from src.networks.noise_layers import ScreenShootingNoiseLayer
from src.predict.loader import create_dataloaders
from src.utils import model_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extracting(
        model: BAGon,
        encoded_dir: str,
        is_noise: bool = True,
        batch_size: int = 2,
        num_workers: int = 2,
        encoded_mapping_file_name: str = "",
        device=device,
):
    decoder = model.decoder
    decoder.eval()

    data_loader = create_dataloaders(
        directory=encoded_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=128
    )

    decoded_messages_mapping = {}
    with torch.inference_mode():
        for batch, (images, image_paths, _) in tqdm(enumerate(data_loader), desc="Decoding",
                                                    total=len(data_loader)):
            images = images.to(device)
            if is_noise:
                noise_layer = ScreenShootingNoiseLayer()
                images = noise_layer(images)

            messages_decoded = decoder(images)
            messages_decoded = messages_decoded.round().clip(0, 1)
            messages_decoded = messages_decoded.detach().cpu().numpy().astype(np.int32)

            for i in range(images.shape[0]):
                image_name = image_paths[i]
                decoded_messages_mapping[image_name] = messages_decoded[i]

    encoded_messages_mapping = {}
    encoded_mapping_file_path = os.path.join(encoded_dir, encoded_mapping_file_name)
    with open(encoded_mapping_file_path, 'r') as f:
        for line in f:
            filename, message_str = line.strip().split()
            message = np.array([int(bit) for bit in message_str])
            encoded_messages_mapping[filename] = message.astype(np.int32)

    assert len(decoded_messages_mapping) == len(encoded_messages_mapping)

    total_bit = 0
    total_wrong_bit = 0
    for key in decoded_messages_mapping:
        encoded_message = encoded_messages_mapping[key]
        decoded_message = decoded_messages_mapping[key]

        wrong_bit = np.sum(np.abs(encoded_message - decoded_message))
        total_wrong_bit += wrong_bit

        if total_bit == 0:
            total_bit = len(encoded_messages_mapping) * len(encoded_message)

    correct_rate = (1 - total_wrong_bit / total_bit) * 100.0
    print(correct_rate)
    return correct_rate, encoded_messages_mapping, decoded_messages_mapping


# def main():
#     model = model_loader.get_model(
#         weights_path='/workspace/code/watermark/bagon/data/result/model/model.pth',
#         device=device
#     )
#     extracting(
#         model=model,
#         encoded_dir="../../data/result/encoded",
#         encoded_mapping_file_name="encoded_message_mapping.txt"
#     )
#
#
# if __name__ == '__main__':
#     main()
