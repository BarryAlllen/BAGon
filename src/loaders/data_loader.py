
from torch.utils.data import DataLoader
from torchvision import transforms

from src.loaders.dataset_custom import TrainDataset, TestDataset


def create_dataloaders(
        train_dir: str,
        test_dir: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        transform: transforms.Compose = None,
        message_matrix_path: str = "./utils/test_matrix.npy"
):
    train_dataset = TrainDataset(
        directory=train_dir,
        image_size=image_size,
        transform=transform
    )

    test_dataset = TestDataset(
        directory=test_dir,
        image_size=image_size,
        message_matrix_path=message_matrix_path,
        transform=transform
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return train_dataloader, test_dataloader

# dataloader,_ = create_dataloaders(
#         train_dir="../../data/train",
#         test_dir="../../data/test",
#         image_size=128,
#         batch_size=8,
#         num_workers=1,
#         message_matrix_path="../utils/test_matrix.npy"
# )
#
# for batch, (a,b,c,d) in enumerate(dataloader):
#     print(batch)
#     print(a)
#     print(b)
#     print(c)
#     print(d)



