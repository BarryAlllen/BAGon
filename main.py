
import os

import argparse

# from torch.backends import cudnn

from src.loaders.data_loader import create_dataloaders
from src.train import Trainer

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main(config):
    # cudnn.benchmark = True

    train_dataloader, test_dataloader = create_dataloaders(
        train_dir=config.train_dir,
        test_dir=config.test_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    engine = Trainer(
        config=config,
        epochs=config.epochs,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        seed=config.seed,
        result_path=config.result_path,
        warmup_steps=config.warmup,
        model_save_step=config.model_save_step,
        epoch_show_step=config.epoch_show_step,
        batch_show_step=config.batch_show_step,
        checkpoint_list=config.checkpoint_list,
        is_wandb=config.is_wandb,
        is_scheduler=config.is_scheduler,
        is_parallel=config.is_parallel,
        is_weights=config.is_weights,
        weights_path=config.weights_path,
        weights_path_discriminator=config.weights_path_discriminator,
    )

    engine.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default="/root/.cache/huggingface/bagon/coco/train_mask_128_10000")
    parser.add_argument("--test_dir", type=str, default="/root/.cache/huggingface/bagon/coco/val2017_1000")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3502)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--model_save_step", type=int, default=10)
    parser.add_argument("--epoch_show_step", type=int, default=1)
    parser.add_argument("--batch_show_step", type=int, default=50)
    parser.add_argument("--checkpoint_list", type=int, nargs='+', default=[55, 65, 75, 85, 95])
    parser.add_argument("--is_wandb", type=bool, default=True)
    parser.add_argument("--is_scheduler", type=bool, default=False)
    parser.add_argument("--is_parallel", type=bool, default=True)
    parser.add_argument("--is_weights", type=bool, default=False)
    path_name = "25Mar16_17h23m36s"
    parser.add_argument("--weights_path", type=str, default=f"/workspace/code/watermark/bagon/data/result/{path_name}/model/model.pth")
    parser.add_argument("--weights_path_discriminator", type=str, default=f"/workspace/code/watermark/bagon/data/result/{path_name}/model/discriminator.pth")
    parser.add_argument("--result_path", type=str, default=f"data/result")
    parser.add_argument("--name", type=str, default=f"barry")

    config = parser.parse_args()
    main(config)