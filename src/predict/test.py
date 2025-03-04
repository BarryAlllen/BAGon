
import os
import torch
import argparse

from src.predict import encoder, decoder
from src.utils import model_loader

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_encoder(config, model):
    encoder.embedding(
        model=model,
        data_dir=config.data_dir,
        output_dir=config.encode_output,
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


def main(config):
    model = model_loader.get_model(
        weights_path=config.model_params_path,
        device=device
    )

    mode_function_mapping = {
        "e": [run_encoder],
        "d": [run_decoder],
        "ed": [run_encoder, run_decoder]
    }

    if config.mode in mode_function_mapping:
        for func in mode_function_mapping[config.mode]:
            func(config, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_params_path", type=str, default="/workspace/code/watermark/bagon/data/result/model/model.pth")
    parser.add_argument("--data_dir", type=str, default="../../data/test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--is_noise", type=bool, default=False)
    parser.add_argument("--encode_output", type=str, default="../../data/result/encoded_noise")
    parser.add_argument("--message_matrix_path", type=str, default="../utils/test_matrix.npy")
    parser.add_argument("--mapping_file_name", type=str, default="encoded_mapping.txt")
    parser.add_argument("--mode", type=str, default="d", choices=['e', 'd', 'ed'], help="Choose a mode: 'e', 'd', or 'ed'")

    config = parser.parse_args()
    print(config)
    main(config)
