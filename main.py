
import os
import argparse

from torch.backends import cudnn

from loaders.data_loader import create_dataloaders
from train import Trainer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(config):
    cudnn.benchmark = True

    train_dataloader, test_dataloader = create_dataloaders(
        train_dir=config.train_dir,
        test_dir=config.test_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    engine = Trainer(
        epochs=config.epochs,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        seed=config.seed,
        is_scheduler=config.is_scheduler,
        is_parallel=config.is_parallel
    )

    engine.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default="./data/train")
    parser.add_argument("--test_dir", type=str, default="./data/test")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3504)
    parser.add_argument("--is_scheduler", type=bool, default=False)
    parser.add_argument("--is_parallel", type=bool, default=True)

    config = parser.parse_args()
    print(config)
    main(config)