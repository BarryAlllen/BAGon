
import os
import cv2
import numpy as np

from typing import Optional

from torch.utils.data import Dataset
from torchvision import transforms

class TrainDataset(Dataset):
    def __init__(self, directory: str, image_size: int, transform: Optional[transforms.Compose] = None):
        super().__init__()
        self.directory = directory
        self.paths = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png'))]
        self.image_size = image_size
        self.transform = transform

    def load_image(self, index: int):
        image_path = self.paths[index]
        image = cv2.imread(os.path.join(self.directory, image_path), 1)
        return image

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        print(index)
        size = self.image_size
        image = self.load_image(index=index)

        if self.transform is not None:
            image = self.transform(image)

        image_origin = image[:, :size, :]
        image_origin = image_origin.transpose((2, 0, 1))
        image_origin = np.float32(image_origin / 255.0 * 2 - 1)

        edge_mask = image[:, size:size * 2, :]
        edge_mask = np.float32(edge_mask) / 255.0
        edge_mask = (edge_mask.transpose((2, 0, 1)) + 1) * 3

        depth_mask = image[:, size * 2:, :]
        depth_mask = np.float32(depth_mask) / 255.0
        depth_mask = (depth_mask.transpose((2, 0, 1)) + 1) * 3

        message = np.random.rand(30)
        message = np.round(message)

        return image_origin, edge_mask, depth_mask, message


class TestDataset(Dataset):
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
        image = self.load_image(index=index)

        if self.transform is not None:
            image = self.transform(image)

        image_test = image[:, :, :]

        image_test = cv2.resize(image_test, (size, size))
        image_test = image_test.transpose((2, 0, 1))
        image_test = np.float32(image_test / 255.0 * 2 - 1)

        message_test = self.message_matrix[index]

        return image_test, message_test