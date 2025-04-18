
import os
import cv2
import numpy as np

from typing import Optional

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class NoiseDataset(Dataset):
    def __init__(self, directory: str, image_size: int, transform: Optional[transforms.Compose] = None):
        super().__init__()
        self.directory = directory
        self.paths = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png'))]
        self.image_size = image_size
        self.transform = transform

    def load_image(self, index: int):
        image_name = self.paths[index]
        image = cv2.imread(os.path.join(self.directory, image_name), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_name

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        size = self.image_size
        image, image_name = self.load_image(index=index)

        _image = image[:, :, :]
        _image = cv2.resize(_image, (size, size))
        _image = _image.transpose((2, 0, 1))
        _image = np.float32(_image / 255.0 * 2 - 1)

        return _image, image_name


def create_noise_dataloaders(
        directory: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        transform: transforms.Compose = None,
):
    noise_dataset = NoiseDataset(
        directory=directory,
        image_size=image_size,
        transform=transform
    )

    noise_dataloader = DataLoader(
        dataset=noise_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return noise_dataloader

# dataloader = create_noise_dataloaders(
#         directory="/workspace/code/watermark/temp_image",
#         image_size=128,
#         batch_size=8,
#         num_workers=1
# )
#
# for batch, (a,b) in enumerate(dataloader):
#     print(batch)
#     print(a)
#     print(b)