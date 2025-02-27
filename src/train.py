
import cv2
import math
import wandb
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.networks.discriminator import DiscriminatorV2 as Discriminator
from src.networks.model import BAGon
from src.utils import check_time


class Trainer:
    def __init__(
            self,
            epochs: int,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            seed: int,
            is_scheduler: bool = False,
            is_parallel: bool = False,
            model_save_path = 'data/result/model/model.pth'
    ):
        self.epochs = epochs

        self.seed = seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.is_parallel = is_parallel
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.is_scheduler = is_scheduler
        self.warmup_steps = 3

        self.alpha = 3
        self.beta = 1
        self.gamma = 0.001

        self.setup_model()
        self.setup_optimizer()
        self.setup_loss_function()
        self.setup_tools()

        self.model_save_path = model_save_path

        self.model_save_step = 10
        self.print_for_epoch = 1
        self.print_for_batch = 25

    def setup_model(self):
        self.bagon_net = BAGon().to(self.device)
        if self.is_parallel:
            self.bagon_net = nn.DataParallel(self.bagon_net)
        self.discriminator = Discriminator(hidden_channels=64).to(self.device)

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step + 0.5) / float(max(1, self.warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * (current_step - self.warmup_steps) / (103 - self.warmup_steps)))

    def setup_optimizer(self):
        self.optimizer = optim.Adam(self.bagon_net.parameters())
        if self.is_scheduler:
            self.scheduler = LambdaLR(self.optimizer, self.lr_lambda)
        self.lr = self.optimizer.param_groups[0]['lr']

        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters())

    def setup_loss_function(self):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def setup_tools(self):
        self.wandb = wandb
        self.wandb.init(
            project="bagon",
            name=check_time.get_time(),
            config={
                "learning_rate": self.lr,
                "architecture": "U-Net",
                "dataset": "COCO",
                "epochs": self.epochs,
            }
        )

    def train(self):

        train_loss = 0.0
        decoder_loss = 0.0
        guide_mask_loss = 0.0
        gen_loss = 0.0
        dis_loss = 0.0

        test_corrcet_total = 0.0
        test_corrcet_best = 0.0

        for epoch in range(self.epochs):
            loss_show  = 0.0
            message_loss_show  = 0.0
            mask_loss_show  = 0.0
            generator_loss_show  = 0.0
            discriminator_fake_loss_show  = 0.0

            print(f"Epoch [{epoch+1}/{self.epochs}] Training begins, lr={self.lr}")
            progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc="Batch training")
            for batch, (image, edge_mask, depth_mask, message) in progress_bar:
                self.bagon_net.train()
                self.discriminator.train()

                image = image.to(self.device).requires_grad_(True)
                edge_mask = edge_mask.to(self.device)
                depth_mask = depth_mask.to(self.device)
                message = message.to(dtype=torch.float).to(self.device)

                image_encoded, image_encoded_noised, message_decoded = self.bagon_net(image, message)  # do forward pass
                message_loss = self.mse_loss(message_decoded, message)

                # get gradient mask
                image_grad = torch.autograd.grad(message_loss, image, create_graph=True)[0]
                gradient_mask = torch.zeros(image_grad.shape, device=self.device)

                for i in range(image_grad.shape[0]):
                    a = image_grad[i, :, :, :]
                    a = (1 - (a - a.min()) / (a.max() - a.min())) + 1
                    gradient_mask[i, :, :, :] = a.detach()

                # train discriminator
                real_labels = torch.full((image.shape[0], 1), 1, dtype=torch.float, device=self.device)
                fake_labels = torch.full((image.shape[0], 1), 0, dtype=torch.float, device=self.device)
                generator_target_labels = torch.full((image.shape[0], 1), 1, dtype=torch.float,
                                                     device=self.device)  # for generator's training

                self.optimizer_discriminator.zero_grad()  # zero_grad for discriminator

                discriminator_real_output = self.discriminator(image.detach())
                discriminator_real_loss = self.bce_loss(discriminator_real_output, real_labels)
                discriminator_real_loss.backward()

                discriminator_fake_output = self.discriminator(image_encoded.detach())
                discriminator_fake_loss = self.bce_loss(discriminator_fake_output, fake_labels)
                discriminator_fake_loss.backward()

                self.optimizer_discriminator.step()

                # train encoder(generator)
                discriminator_generator_output = self.discriminator(image_encoded)
                generator_loss = self.bce_loss(discriminator_generator_output, generator_target_labels)

                message_loss = self.mse_loss(message_decoded, message)

                # guide to watermark embedding position
                mask_loss = (
                        self.mse_loss(image_encoded * depth_mask.float(), image * depth_mask.float()) * 0.75 +
                        self.mse_loss(image_encoded * gradient_mask.float(), image * gradient_mask.float()) * 0.5 +
                        self.mse_loss(image_encoded * edge_mask.float(), image * edge_mask.float()) * 2
                )

                loss = (
                        message_loss * self.alpha +
                        mask_loss * self.beta +
                        generator_loss * self.gamma
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.lr = self.optimizer.param_groups[0]['lr']

                loss_show += loss.item()
                message_loss_show += message_loss.item()
                mask_loss_show += mask_loss.item()
                generator_loss_show += generator_loss.item()
                discriminator_fake_loss_show += discriminator_fake_loss.item()

                if (batch + 1) % self.print_for_batch == 0:
                    print(f"\nEpoch [{epoch + 1}/{self.epochs}], Batch [{batch + 1}/{len(self.train_dataloader)}], "
                          f"Loss: {loss_show / self.print_for_batch}, "
                          f"Message Loss: {message_loss_show / self.print_for_batch}, "
                          f"Mask Loss: {mask_loss_show / self.print_for_batch}, "
                          f"Generator Loss: {generator_loss_show / self.print_for_batch}, "
                          f"Discriminator Fake Loss: {discriminator_fake_loss_show / self.print_for_batch}")

                    # self.wandb.log({
                    #     "Loss": loss_show / self.print_for_batch,
                    #     "Message Loss": message_loss_show / self.print_for_batch,
                    #     "Mask Loss": mask_loss_show / self.print_for_batch,
                    #     "Generator Loss": generator_loss_show / self.print_for_batch,
                    #     "Discriminator Fake Loss": discriminator_fake_loss_show / self.print_for_batch,
                    #     "lr": self.lr
                    # },step=batch)

                    loss_show = 0.0
                    message_loss_show = 0.0
                    mask_loss_show = 0.0
                    generator_loss_show = 0.0
                    discriminator_fake_loss_show = 0.0

                    self.save_training_image(image, image_encoded, image_encoded_noised, gradient_mask, edge_mask, depth_mask, epoch, batch)

                train_loss += loss.item()
                decoder_loss += message_loss.item()
                guide_mask_loss += mask_loss.item()
                gen_loss += generator_loss.item()
                dis_loss += discriminator_fake_loss.item()

            self.wandb.log({
                "Train Loss": train_loss / len(self.train_dataloader),
                "Decoder Loss": decoder_loss / len(self.train_dataloader),
                "Guide Mask Loss": guide_mask_loss / len(self.train_dataloader),
                "Generator Loss": gen_loss / len(self.train_dataloader),
                "Discriminator Fake Loss": dis_loss / len(self.train_dataloader),
                "lr": self.lr
            })

            train_loss = 0.0
            decoder_loss = 0.0
            guide_mask_loss = 0.0
            gen_loss = 0.0
            dis_loss = 0.0

            if self.is_scheduler:
                self.scheduler.step()

            wrong_correct_bit = 0.0
            correct_bit_total = 0.0

            print('Testing begins...')
            self.bagon_net.eval()
            with torch.inference_mode():
                for batch, (image_test, message_test) in enumerate(self.test_dataloader):
                    image_test = image_test.to(self.device)
                    message_test = message_test.to(dtype=torch.float).to(self.device)

                    image_encoded_test, _, message_decoded_test = self.bagon_net(image_test, message_test)
                    message_test = message_test.to(dtype=torch.int)
                    message_decoded_test = torch.round(message_decoded_test).to(dtype=torch.int)
                    wrong_correct_bit += torch.sum(torch.abs(message_decoded_test - message_test)).item()
                    correct_bit_total += message_test.shape[0] * message_test.shape[1]

            test_correct = (1 - wrong_correct_bit / correct_bit_total) * 100.0
            print(f"Epoch [{epoch + 1}/{self.epochs}] Correct Rate: {test_correct:.2f}%\n")
            self.wandb.log({
                "Correct Rate": test_correct,
            })

            if test_correct >= test_corrcet_best:
                test_corrcet_best = test_correct

            test_corrcet_total += test_correct

            if (epoch + 1) % self.model_save_step == 0:
                torch.save(self.bagon_net.state_dict(), self.model_save_path)

        self.wandb.finish()

    def save_training_image(self, image, image_encoded, image_encoded_noised, gradient_mask, edge_mask, depth_mask, epoch, batch):
        t = np.random.randint(image.shape[0])
        img_1 = (image[t, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
        img_1 = np.transpose(img_1, (1, 2, 0))

        img_2 = (image_encoded[t, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
        img_2 = np.transpose(img_2, (1, 2, 0))

        img_noised = (image_encoded_noised[t, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
        img_noised = np.transpose(img_noised, (1, 2, 0))

        img_gradient_mask = (gradient_mask[t, :, :, :].detach().to('cpu').numpy() - 1) * 255
        img_gradient_mask = np.transpose(img_gradient_mask, (1, 2, 0))

        img_edge_mask = (edge_mask[t, :, :, :].detach().to('cpu').numpy() - 1) * 50
        img_edge_mask = np.transpose(img_edge_mask, (1, 2, 0))

        img_depth_mask = (depth_mask[t, :, :, :].detach().to('cpu').numpy() - 1) * 50
        img_depth_mask = np.transpose(img_depth_mask, (1, 2, 0))

        img_residual = (img_2 - img_1) * 5

        result = np.zeros((img_1.shape[0], img_1.shape[1] * 7, img_1.shape[2]))

        shape = img_1.shape[1]
        result[:, :shape, :] = img_1
        result[:, shape * 1:shape * 2, :] = img_2
        result[:, shape * 2:shape * 3, :] = img_noised
        result[:, shape * 3:shape * 4, :] = img_gradient_mask
        result[:, shape * 4:shape * 5, :] = img_edge_mask
        result[:, shape * 5:shape * 6, :] = img_depth_mask
        result[:, shape * 6:shape * 7, :] = img_residual

        cv2.imwrite(f'data/result/epoch_batch_image/epoch{epoch+1}-batch{batch+1}.png', result)

