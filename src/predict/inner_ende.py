import torch
from torch.utils.data import DataLoader

from src.loaders.dataset_custom import TestDataset
from src.networks.model import BAGon

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inner_test(
        model: BAGon,
        data_dir: str,
        is_noise: bool = True,
        image_size: int = 128,
        batch_size: int = 2,
        num_workers: int = 1,
        message_matrix_path: str = "../utils/test_matrix.npy",
        device=device,
):
    test_dataset = TestDataset(
        directory=data_dir,
        image_size=image_size,
        message_matrix_path=message_matrix_path
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    wrong_correct_bit = 0.0
    correct_bit_total = 0.0
    model.eval()
    with torch.inference_mode():
        for batch, (image_test, message_test) in enumerate(test_dataloader):
            image_test = image_test.to(device)
            message_test = message_test.to(dtype=torch.float).to(device)
            image_encoded_test = model.encoder(image_test, message_test)
            if is_noise:
                print("Noise")
                image_encoded_test = model.noise_layer(image_encoded_test)
            message_decoded_test = model.decoder(image_encoded_test)
            message_test = message_test.to(dtype=torch.int)
            message_decoded_test = torch.round(message_decoded_test).to(dtype=torch.int)
            wrong_correct_bit += torch.sum(torch.abs(message_decoded_test - message_test)).item()
            correct_bit_total += message_test.shape[0] * message_test.shape[1]

    test_correct = (1 - wrong_correct_bit / correct_bit_total) * 100.0
    print(test_correct)
