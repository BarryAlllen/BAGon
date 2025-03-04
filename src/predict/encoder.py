import os

import cv2
import numpy as np
import torch

from tqdm import tqdm

from src.networks.model import BAGon
from src.predict.loader import create_dataloaders
from src.utils import model_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def embedding(
        model: BAGon,
        data_dir: str,
        output_dir: str,
        image_size: int = 128,
        batch_size: int = 2,
        num_workers: int = 1,
        message_matrix_path: str = "../utils/test_matrix.npy",
        mapping_file_name:str = "encoded_mapping.txt",
        device = device,
):
    encoder = model.encoder
    encoder.eval()

    data_loader = create_dataloaders(
        directory=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        message_matrix_path=message_matrix_path
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mapping_path = os.path.join(output_dir, mapping_file_name)

    with open(mapping_path, 'w') as f:
        with torch.inference_mode():
            for batch, (images, image_paths, messages) in tqdm(enumerate(data_loader), desc="Encoding",
                                                               total=len(data_loader)):
                images = images.to(device)
                messages = messages.to(dtype=torch.float).to(device)

                images_encoded = encoder(images, messages)
                images_encoded = (images_encoded.detach() + 1) / 2 * 255

                for i in range(images_encoded.shape[0]):
                    image_encoded = images_encoded[i]
                    image_encoded = image_encoded.permute(1, 2, 0)
                    image_encoded = image_encoded.cpu().numpy()

                    image_path = image_paths[i].split(".")[0]
                    output_name = image_path + "_encoded.jpg"

                    output = os.path.join(output_dir, output_name)
                    cv2.imwrite(output, image_encoded)

                    message = messages[i].cpu().numpy() if isinstance(messages[i], torch.Tensor) else messages[i]
                    message_str = ''.join(map(str, message.astype(int))) if isinstance(message, np.ndarray) else str(message)
                    f.write(f"{output_name} {message_str}\n")

# def main():
#     model = model_loader.get_model(
#         weights_path='/workspace/code/watermark/bagon/data/result/model/model.pth',
#         device=device
#     )
#     embedding(
#         model=model,
#         data_dir="../../data/test",
#         output_dir='../../data/result/encoded'
#     )
#
#
# if __name__ == '__main__':
#     main()
