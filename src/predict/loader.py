
import os
import cv2
import numpy as np

from typing import Optional

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PredictDataset(Dataset):
    def __init__(self, directory: str, image_size: int, message_matrix_path: str, transform: Optional[transforms.Compose] = None):
        super().__init__()
        self.directory = directory
        self.paths = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png'))]
        self.image_size = image_size
        self.message_matrix = np.load(message_matrix_path)
        self.transform = transform

    def load_image(self, index: int):
        image_path = self.paths[index]
        image = cv2.imread(os.path.join(self.directory, image_path), 1)
        return image

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        size = self.image_size
        image_path = self.paths[index]
        image = self.load_image(index=index)

        image_predict = image[:, :, :]
        image_predict = cv2.resize(image_predict, (size, size))
        image_predict = image_predict.transpose((2, 0, 1))
        image_predict = np.float32(image_predict / 255.0 * 2 - 1)

        message = self.message_matrix[index]

        return image_predict, image_path, message


def create_dataloaders(
        directory: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        transform: transforms.Compose = None,
        message_matrix_path: str = "../utils/test_matrix.npy"
):
    predict_dataset = PredictDataset(
        directory=directory,
        image_size=image_size,
        message_matrix_path=message_matrix_path,
        transform=transform
    )

    predict_dataloader = DataLoader(
        dataset=predict_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return predict_dataloader

# dataloader = create_dataloaders(
#         directory="../../data/train",
#         image_size=128,
#         batch_size=8,
#         num_workers=1
# )
#
# for batch, (a,b,c) in enumerate(dataloader):
#     print(batch)
#     print(a)
#     print(b)
#     print(c)