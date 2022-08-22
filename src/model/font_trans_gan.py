from pathlib import Path
import torch
from torch.nn import L1Loss, DataParallel
from torch.optim import Adam, lr_scheduler
from src.config import (
    BETA1,
    DEVICE,
    LAMBDA_CONTENT,
    LAMBDA_L1,
    LAMBDA_STYLE,
    LR,
    STYLE_CHANNEL,
)
from src.model.utils import init_net, lambda_rule

from src.model.generator import FTGAN_Generator_MLAN
from src.model.discriminator import NLayerDiscriminatorS

from src.model.loss import GANLoss


class FTGAN:
    def __init__(
        self,
        device=DEVICE,
        style_channel=STYLE_CHANNEL,
        lambda_style=LAMBDA_STYLE,
        lambda_content=LAMBDA_CONTENT,
        lambda_L1=LAMBDA_L1,
        lr=LR,
        beta1=BETA1,
        is_train=True,
    ):
        self.device = device
        self.style_channel = style_channel

        # Networks definition
        self.netG = FTGAN_Generator_MLAN()
        self.netG = init_net(self.netG, device)

        self.netD_content = NLayerDiscriminatorS(2)
        self.netD_content = init_net(self.netD_content, device)

        self.netD_style = NLayerDiscriminatorS(style_channel + 1)
        self.netD_style = init_net(self.netD_style, device)

        # Loss settings
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        self.lambda_L1 = lambda_L1
        self.criterionGAN = GANLoss("hinge")
        self.criterionL1 = L1Loss()

        # Optimizers and schedulers for each network
        self.optimizer_G = Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.scheduler_G = lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=lambda_rule
        )

        self.optimizer_D_content = Adam(
            self.netD_content.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        self.scheduler_D_content = lr_scheduler.LambdaLR(
            self.optimizer_D_content, lr_lambda=lambda_rule
        )

        self.optimizer_D_style = Adam(
            self.netD_style.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        self.scheduler_D_style = lr_scheduler.LambdaLR(
            self.optimizer_D_style, lr_lambda=lambda_rule
        )

    def _compute_gan_loss_D(self, real_images, fake_images, netD):
        # Fake
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real = torch.cat(real_images, 1)
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    def _get_loss_D_content_style(self, content_img, style_img, gt_img, generated_img):
        loss_D_content = self._compute_gan_loss_D(
            [content_img, gt_img],
            [content_img, generated_img],
            self.netD_content,
        )
        loss_D_style = self._compute_gan_loss_D(
            [style_img, gt_img],
            [style_img, generated_img],
            self.netD_style,
        )
        loss_D = self.lambda_content * loss_D_content + self.lambda_style * loss_D_style
        return loss_D

    def _compute_gan_loss_G(self, fake_images, netD):
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake)
        loss_G_GAN = self.criterionGAN(pred_fake, True, True)
        return loss_G_GAN

    def _get_loss_G(self, content_img, style_img, gt_img, generated_img):
        loss_G_content = self._compute_gan_loss_G(
            [content_img, generated_img], self.netD_content
        )
        loss_G_style = self._compute_gan_loss_G(
            [style_img, generated_img], self.netD_style
        )
        loss_G_GAN = (
            self.lambda_content * loss_G_content + self.lambda_style * loss_G_style
        )

        loss_G_L1 = self.criterionL1(generated_img, gt_img) * self.lambda_L1

        loss_G = loss_G_GAN + loss_G_L1

        return loss_G

    def _set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def train_D_content_style(self, content_img, style_img, gt_img, generated_img):
        self._set_requires_grad(self.netD_content, True)
        self._set_requires_grad(self.netD_style, True)

        self.optimizer_D_content.zero_grad()
        self.optimizer_D_style.zero_grad()

        loss_D = self._get_loss_D_content_style(
            content_img, style_img, gt_img, generated_img
        )
        loss_D.backward()

        self.optimizer_D_content.step()
        self.optimizer_D_style.step()

        self.scheduler_D_content.step()
        self.scheduler_D_style.step()

        self._set_requires_grad(self.netD_content, False)
        self._set_requires_grad(self.netD_style, False)

    def train_G(self, content_img, style_img, gt_img, generated_img):
        self.optimizer_G.zero_grad()

        loss_G = self._get_loss_G(content_img, style_img, gt_img, generated_img)
        loss_G.backward()

        self.optimizer_G.step()
        self.scheduler_G.step()

    def train_step(self, content_img, style_img, gt_img):
        generated_img = self.netG((content_img, style_img))

        self.train_D_content_style(content_img, style_img, gt_img, generated_img)
        self.train_G(content_img, style_img, gt_img, generated_img)

    def compute_visuals(self):
        pass

    def load_checkpoint(self, checkpoint_folder: Path, prefix: str):
        checkpoint_path = checkpoint_folder / f"{prefix}_net_G.pth"
        self.netG.module.load_state_dict(torch.load(checkpoint_path))

        checkpoint_path = checkpoint_folder / f"{prefix}_net_D_content.pth"
        self.netD_content.module.load_state_dict(torch.load(checkpoint_path))

        checkpoint_path = checkpoint_folder / f"{prefix}_net_D_style.pth"
        self.netD_style.module.load_state_dict(torch.load(checkpoint_path))
