import torch
from torch.nn import Module, L1Loss
from torch.optim import Adam
from src.constants import BETA1, LR, NORM_LAYER, STYLE_CHANNEL

from src.model.generator import FTGAN_Generator_MLAN
from src.model.discriminator import NLayerDiscriminatorS

from src.loss.gan_loss import GANLoss

class FTGAN_Train(Module):
    def __init__(self, style_channel=STYLE_CHANNEL, lr=LR, beta1=BETA1, is_train=True):
        super(FTGAN_Train, self).__init__()

        self.style_channel = style_channel

        self.netG = FTGAN_Generator_MLAN(norm_layer=NORM_LAYER)
        self.netD_content = NLayerDiscriminatorS(2, norm_layer=NORM_LAYER)
        self.netD_style = NLayerDiscriminatorS(style_channel+1, norm_layer=NORM_LAYER)

        self.lambda_style = 1
        self.lambda_content = 1
        self.lamda_L1 = 100
        self.criterionGAN = GANLoss('hinge')
        self.criterionL1 = L1Loss()

        self.optimizer_G = Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D_content = Adam(self.netD_content.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D_style = Adam(self.netD_style.parameters(), lr=lr, betas=(beta1, 0.999))

    def _forward(self, content, style):
        return self.netG((content, style))

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

    def _backward_D(self, content_img, style_img, gt_img, generated_img):
        loss_D_content = self.compute_gan_loss_D(
            [content_img, gt_img],
            [content_img, generated_img],
            self.netD_content,
        )
        loss_D_style = self.compute_gan_loss_D(
            [style_img, gt_img],
            [style_img, generated_img],
            self.netD_style,
        )
        loss_D = (
            self.lambda_content * loss_D_content
            + self.lambda_style * loss_D_style
        )
        loss_D.backward()
        return loss_D

    def _compute_gan_loss_G(self, fake_images, netD):
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake)
        loss_G_GAN = self.criterionGAN(pred_fake, True, True)
        return loss_G_GAN

    def _backward_G(self, content_img, style_img, gt_img, generated_img):
        loss_G_content = self.compute_gan_loss_G(
            [content_img, generated_img], self.netD_content
        )
        loss_G_style = self.compute_gan_loss_G(
            [style_img, generated_img], self.netD_style
        )
        loss_G_GAN = (
            self.lambda_content * loss_G_content
            + self.lambda_style * loss_G_style
        )

        loss_G_L1 = (
            self.criterionL1(generated_img, gt_img) * self.opt.lambda_L1
        )

        loss_G = loss_G_GAN + loss_G_L1

        loss_G.backward()
        return loss_G


    def train_step(self, content_img, style_img, gt_img):
        generated_img = self.forward(content_img, style_img)
        # set requires_grad=True for all params
        self.optimizer_D_content.zero_grad()
        self.optimizer_D_style.zero_grad()
        self._backward_D(content_img, style_img, gt_img, generated_img)
        self.optimizer_D_content.step()
        self.optimizer_D_style.step()
        # set requires_grad=False for all params

        self.optimizer_G.zero_grad()
        self._backward_G(content_img, style_img, gt_img, generated_img)
        self.optimizer_G.step()

    def compute_visuals(self):
        pass

    def set_up(self):
        # create scheduler for optimizer
        # linear
        pass