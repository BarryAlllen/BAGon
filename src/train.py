import os
import sys

import cv2
import math
import wandb
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from loguru import logger

from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.networks.discriminator import Discriminator as Discriminator
from src.networks.model import BAGon
from src.utils import check_time as ct


class Trainer:
    def __init__(
            self,
            config: dict,
            epochs: int,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            seed: int,
            result_path: str,
            warmup_steps: int,
            model_save_step: int,
            epoch_show_step: int,
            batch_show_step: int,
            is_wandb: bool,
            is_scheduler: bool = False,
            is_parallel: bool = False,
    ):
        self.config = config
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
        self.warmup_steps = warmup_steps

        self.alpha = 3
        self.beta = 1
        self.gamma = 0.001

        time = ct.get_main_time()

        self.setup_result_path(time=time, path=result_path)
        self.setup_logger(time=time, path=result_path)
        self.setup_model()
        self.setup_optimizer()
        self.setup_loss_function()
        self.is_wandb = is_wandb
        self.setup_tools(time=time)

        self.model_save_step = model_save_step
        self.print_for_epoch = epoch_show_step
        self.print_for_batch = batch_show_step

    def setup_model(self):
        self.bagon_net = BAGon().to(self.device)
        logger.info(f"{self.bagon_net.encoder.name}")
        logger.info(f"{self.bagon_net.decoder.name}")
        if self.is_parallel:
            self.bagon_net = nn.DataParallel(self.bagon_net)

        self.discriminator = Discriminator(hidden_channels=64).to(self.device)
        logger.info(f"{self.discriminator.name}")
        if self.is_parallel:
            self.discriminator = nn.DataParallel(self.discriminator)

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step + 0.5) / float(max(1, self.warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * (current_step - self.warmup_steps) / (103 - self.warmup_steps)))

    def setup_optimizer(self):
        self.optimizer = optim.Adam(self.bagon_net.parameters())
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters())

        if self.is_scheduler:
            self.scheduler = LambdaLR(self.optimizer, self.lr_lambda)
            self.scheduler_discriminator = LambdaLR(self.optimizer_discriminator, self.lr_lambda)
        self.lr = self.optimizer.param_groups[0]['lr']
        self.lr_discriminator = self.optimizer_discriminator.param_groups[0]['lr']

    def setup_loss_function(self):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def setup_tools(self, time):
        if self.is_wandb:
            self.wandb = wandb
            self.wandb.init(
                project="bagon",
                name=time,
                config={
                    "learning_rate": self.lr,
                    "architecture": "U-Net",
                    "dataset": "COCO",
                    "epochs": self.epochs,
                }
            )

    def setup_result_path(self, time: str, path: str):
        # model
        model_path = os.path.join(path, f"{time}/model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_save_path = model_path

        # snapshot
        self.snapshot_path = os.path.join(path, f"{time}/snapshot")
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

    def setup_logger(self, time: str, path: str):
        logger.remove()
        log_file = "train.log"
        log_path = os.path.join(path, f"{time}")
        logger.add(
            os.path.join(log_path, log_file),
            format="<level>{message}</level>"
        )
        logger.add(sys.stdout, format="<level>{message}</level>")
        logger.info(self.config)

    def train(self):
        start_time = ct.get_time()  # record start time
        logger.info(f"Start time: {ct.get_time(format=ct.time_format1)}")

        train_loss = 0.0
        decoder_loss = 0.0
        guide_mask_loss = 0.0
        gen_loss = 0.0
        dis_loss = 0.0

        test_corrcet_total = 0.0
        test_corrcet_best = 0.0

        for epoch in range(self.epochs):
            loss_show = 0.0
            message_loss_show = 0.0
            mask_loss_show = 0.0
            generator_loss_show = 0.0
            discriminator_fake_loss_show = 0.0

            logger.info(
                f"Epoch [{epoch + 1}/{self.epochs}] Training begins, lr={self.lr:.10f}, lr_discriminator={self.lr_discriminator:.10f}")
            progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader),
                                desc="Batch training")
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
                        self.mse_loss(image_encoded * depth_mask.float(), image * depth_mask.float()) * 2 +
                        self.mse_loss(image_encoded * edge_mask.float(), image * edge_mask.float()) * 0.75 +
                        self.mse_loss(image_encoded * gradient_mask.float(), image * gradient_mask.float()) * 0.25
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
                self.lr_discriminator = self.optimizer_discriminator.param_groups[0]['lr']

                loss_show += loss.item()
                message_loss_show += message_loss.item()
                mask_loss_show += mask_loss.item()
                generator_loss_show += generator_loss.item()
                discriminator_fake_loss_show += discriminator_fake_loss.item()

                if (batch + 1) % self.print_for_batch == 0:
                    logger.info(f"\nEpoch [{epoch + 1}/{self.epochs}], Batch [{batch + 1}/{len(self.train_dataloader)}], "
                          f"Loss: {loss_show / self.print_for_batch:.4f}, "
                          f"Message Loss: {message_loss_show / self.print_for_batch:.4f}, "
                          f"Mask Loss: {mask_loss_show / self.print_for_batch:.4f}, "
                          f"Generator Loss: {generator_loss_show / self.print_for_batch:.4f}, "
                          f"Discriminator Fake Loss: {discriminator_fake_loss_show / self.print_for_batch:.4f}")

                    # if self.is_wandb:
                    #     self.wandb.log({
                    #         "Loss": loss_show / self.print_for_batch,
                    #         "Message Loss": message_loss_show / self.print_for_batch,
                    #         "Mask Loss": mask_loss_show / self.print_for_batch,
                    #         "Generator Loss": generator_loss_show / self.print_for_batch,
                    #         "Discriminator Fake Loss": discriminator_fake_loss_show / self.print_for_batch,
                    #         "lr": self.lr
                    #     },step=batch)

                    loss_show = 0.0
                    message_loss_show = 0.0
                    mask_loss_show = 0.0
                    generator_loss_show = 0.0
                    discriminator_fake_loss_show = 0.0

                    self.save_training_image(image, image_encoded, image_encoded_noised, gradient_mask, edge_mask,
                                             depth_mask, epoch, batch)

                train_loss += loss.item()
                decoder_loss += message_loss.item()
                guide_mask_loss += mask_loss.item()
                gen_loss += generator_loss.item()
                dis_loss += discriminator_fake_loss.item()
            if self.is_wandb:
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
                self.scheduler_discriminator.step()

            wrong_correct_bit = 0.0
            correct_bit_total = 0.0

            logger.info('Testing begins...')
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
            logger.info(f"Epoch [{epoch + 1}/{self.epochs}] Correct Rate: {test_correct:.2f}% | {ct.get_time(format=ct.time_format3)}\n")
            if self.is_wandb:
                self.wandb.log({
                    "Correct Rate": test_correct,
                })

            # save best model weights
            filename = "model.pth" # model base name
            if test_correct >= test_corrcet_best:
                test_corrcet_best = test_correct

                # save best encoder & decoder
                best_model_save_path = os.path.join(self.model_save_path, f"best_{filename}")
                torch.save(self.bagon_net.state_dict(), best_model_save_path)

                # save best discriminator
                best_discriminator_save_path = os.path.join(self.model_save_path, f"best_discriminator_{filename}")
                torch.save(self.discriminator.state_dict(), best_discriminator_save_path)

                logger.info(f"Best model saved at {self.model_save_path}\n")

            test_corrcet_total += test_correct

            # save model weights
            if (epoch + 1) % self.model_save_step == 0:
                # save encoder & decoder
                step_model_save_path = os.path.join(self.model_save_path, filename)
                torch.save(self.bagon_net.state_dict(), step_model_save_path)

                # save discriminator
                step_discriminator_save_path = os.path.join(self.model_save_path, f"discriminator_{filename}")
                torch.save(self.discriminator.state_dict(), step_discriminator_save_path)

                logger.info(f"Model saved at {self.model_save_path}, epoch {epoch + 1}\n")

        end_time = ct.get_time()  # record end time
        logger.info(f"End time: {ct.get_time(format=ct.time_format1)}")
        time_diff = ct.cal_time_diff(start_time, end_time)
        logger.info(f"Training finished, total time: {time_diff}")

        if self.is_wandb:
            self.wandb.finish()

    def save_training_image(self, image, image_encoded, image_encoded_noised, gradient_mask, edge_mask, depth_mask,
                            epoch, batch):
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

        image_snapshot = os.path.join(self.snapshot_path, f"e{epoch + 1}-b{batch + 1}.png")
        cv2.imwrite(filename=image_snapshot, img=result)
