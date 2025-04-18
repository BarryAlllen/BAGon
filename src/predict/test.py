
import os
import torch
import argparse

from torch.utils.data import DataLoader

from src.loaders.dataset_custom import TestDataset
from src.predict import encoder, decoder, inner_ende
from src.utils import model_loader

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_encoder(config, model):
    encoder.embedding(
        model=model,
        data_dir=config.data_dir,
        output_dir=config.encode_output,
        clean_dir=config.clean_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        message_matrix_path=config.message_matrix_path,
        mapping_file_name=config.mapping_file_name,
        device=device
    )


def run_decoder(config, model):
    decoder.extracting(
        model=model,
        encoded_dir=config.encode_output,
        is_noise=config.is_noise,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        encoded_mapping_file_name=config.mapping_file_name,
        device=device
    )


def run_inner_encoder_decoder(config, model):
    inner_ende.inner_test(
        model=model,
        data_dir=config.data_dir,
        is_noise=config.is_noise,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        message_matrix_path=config.message_matrix_path,
        device=device
    )


def main(config):
    model = model_loader.get_model(
        weights_path=config.model_params_path,
        device=device
    )

    mode_function_mapping = {
        "e": [run_encoder],
        "d": [run_decoder],
        "ed": [run_encoder, run_decoder],
        "i": [run_inner_encoder_decoder]
    }

    if config.mode in mode_function_mapping:
        for func in mode_function_mapping[config.mode]:
            func(config, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path_name = "25Mar16_17h23m36s"
    path_name = "25Mar5_15h06m25s"
    parser.add_argument("--model_params_path", type=str, default=f"/workspace/code/watermark/bagon/data/result/{path_name}/model/model.pth")
    # parser.add_argument("--data_dir", type=str, default="../../data/result/benchmark/image")
    # parser.add_argument("--data_dir", type=str, default="/root/.cache/huggingface/coco/coco/bagon_coco2017_val_1000")
    parser.add_argument("--data_dir", type=str, default="/workspace/code/watermark/temp_image")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=56)
    parser.add_argument("--is_noise", type=bool, default=True)
    parser.add_argument("--is_jpg", type=bool, default=True)
    parser.add_argument("--clean_dir", type=bool, default=True)
    # parser.add_argument("--encode_output", type=str, default="../../data/result/benchmark/encoded")
    parser.add_argument("--encode_output", type=str, default="/workspace/code/watermark/temp_encoded_v0")
    parser.add_argument("--message_matrix_path", type=str, default="../utils/test_matrix.npy")
    parser.add_argument("--mapping_file_name", type=str, default="encoded_mapping.txt")
    parser.add_argument("--mode", type=str, default="ed", choices=['e', 'd', 'ed', 'i'], help="Choose a mode: 'e', 'd', or 'ed'")

    config = parser.parse_args()
    print(config)
    main(config)
