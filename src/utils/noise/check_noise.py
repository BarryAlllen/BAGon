import os
from PIL import Image
import torch
import numpy as np
from typing import List, Dict

from src.utils.noise.loader import create_noise_dataloaders
# 假设这些噪声类已经定义在 src.utils.noise.noise_layers 中
from src.utils.noise.noise_layers import JPEGNoise, PerspectiveNoise, LightNoise, MoireNoise, GaussianNoise


class NoiseApplier:
    def __init__(self, noise_types: List[str], device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_models = self._init_noise_models(noise_types)

    def _init_noise_models(self, noise_types: List[str]) -> Dict[str, torch.nn.Module]:
        noise_mapping = {
            "jpg": JPEGNoise(),
            "perspective": PerspectiveNoise(),
            "light": LightNoise(),
            "moire": MoireNoise(),
            "gaussian": GaussianNoise()
        }

        models = {}
        for noise_type in noise_types:
            if noise_type in noise_mapping:
                models[noise_type] = noise_mapping[noise_type].to(self.device)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
        return models

    def apply(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        results = {}
        image_tensor = image_tensor.to(self.device)

        for noise_type, model in self.noise_models.items():
            with torch.no_grad():
                results[noise_type] = model(image_tensor)
        return results


def save_noised_images(noised_images: Dict[str, torch.Tensor],
                       image_names: List[str],
                       output_dirs: Dict[str, str]):
    """
    最终版RGB保存函数
    """
    for noise_type, batch_tensor in noised_images.items():
        images = batch_tensor.cpu().numpy()
        images = (images + 1.0) / 2.0 * 255
        images = images.clip(0, 255).astype('uint8')
        images = images.transpose(0, 2, 3, 1)  # BCHW -> BHWC

        for i, image in enumerate(images):
            # 确保3通道
            if image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=2)

            # 保存为RGB
            output_path = os.path.join(output_dirs[noise_type], image_names[i])
            Image.fromarray(image).save(output_path)


def process_images(
        image_dir: str,
        noise_types: List[str],
        image_size: int = 256,
        batch_size: int = 1,
        num_workers: int = 4,
        device: torch.device = None
):
    # 创建输出目录
    output_dirs = {}
    for noise_type in noise_types:
        output_dir = os.path.join(image_dir, f"{noise_type}_noised")
        os.makedirs(output_dir, exist_ok=True)
        output_dirs[noise_type] = output_dir

    # 初始化噪声应用器
    noise_applier = NoiseApplier(noise_types, device)

    # 创建数据加载器
    dataloader = create_noise_dataloaders(
        directory=image_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 处理每个批次
    for batch in dataloader:
        images, image_names = batch
        images = images.to(device)

        # 应用噪声
        noised_images = noise_applier.apply(images)

        # 保存结果
        save_noised_images(noised_images, image_names, output_dirs)

        print(f"Processed batch with images: {', '.join(image_names)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--image_dir", type=str, default="/workspace/code/watermark/temp_image")
    parser.add_argument("--noise_types", nargs="+", default=["jpg", "perspective", "light", "moire", "gaussian"],
                        choices=["jpg", "perspective", "light", "moire", "gaussian"])
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    args = parser.parse_args()

    device = None
    if args.device:
        device = torch.device(args.device)

    process_images(
        image_dir=args.image_dir,
        noise_types=args.noise_types,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device
    )