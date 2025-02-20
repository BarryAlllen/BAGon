
import os
import cv2
import numpy as np

from typing import Optional
from torch.utils import data
from torchvision import transforms


class DataLoader(data.Dataset):
    def __init__(self, directory: str, image_size: int, transform: Optional[transforms.Compose] = None):
        super().__init__()
        self.directory = directory
        self.paths = os.listdir(directory)
        self.transform = transform
        self.image_size = image_size

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
        message[message >= 0.5] = 1
        message[message < 0.5] = 0

        return image_origin, edge_mask, depth_mask, message