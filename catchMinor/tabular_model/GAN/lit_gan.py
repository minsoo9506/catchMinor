import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from catchMinor.tabular_model.GAN.gan_config import (
    GAN_config,
    GAN_loss_func_config,
    GAN_optimizer_config,
)
from catchMinor.tabular_model.GAN.torch_gan import Discriminator, Generator
from catchMinor.utils.debug import get_logger


class LitGAN(pl.LightningModule):
    def __init__(
        self,
        model_config: GAN_config,
        optimizer_config: GAN_loss_func_config,
        loss_func_config: GAN_optimizer_config,
    ):
        super().__init__()
        self._latent_dim = model_config.generator_dim_list[0]
        self.automatic_optimization = False

        self.generator = Generator(model_config)
        self.discriminator = Discriminator(model_config)
        self.optimizer_config = optimizer_config
        self.loss_func_config = loss_func_config
        logger = get_logger(logger_setLevel="INFO")
        logger.info("GAN with fully-connected layer is made.")

    def forward(self, z):
        return self.generator(z)

    def _configure_loss_func(self):
        pass

    def _adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        # optimizer
        optimizer_g, optimizer_d = self.optimizers()
        # sample noise
        z = torch.randn(x.shape[0], self._latent_dim)
        z = z.type_as(x)
        # train generator
        self.toggle_optimizer(optimizer_g)
        # generate images
        self.generated_imgs = self.generator(z)
        # ground truth result (ie: all fake)
        real = torch.ones(x.size(0), 1)
        real = real.type_as(x)
        # binary cross-entropy
        g_loss = self._adversarial_loss(self.discriminator(self.generator(z)), real)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        # train discriminator
        self.toggle_optimizer(optimizer_d)
        real = torch.ones(x.size(0), 1)
        real = real.type_as(x)
        real_loss = self.adversarial_loss(self.discriminator(x), real)

        fake = torch.zeros(x.size(0), 1)
        fake = fake.type_as(x)
        fake_loss = self.adversarial_loss(
            self.discriminator(self.generator(z).detach()), fake
        )
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z = torch.randn(x.shape[0], self._latent_dim)
        z = z.type_as(x)
        self.generated_imgs = self.generator(z)

        real = torch.ones(x.size(0), 1)
        real = real.type_as(x)
        g_loss = self._adversarial_loss(self.discriminator(self.generator(z)), real)

        real = torch.ones(x.size(0), 1)
        real = real.type_as(x)
        real_loss = self._adversarial_loss(self.discriminator(x), real)

        fake = torch.zeros(x.size(0), 1)
        fake = fake.type_as(x)
        fake_loss = self._adversarial_loss(
            self.discriminator(self.generator(z).detach()), fake
        )
        d_loss = (real_loss + fake_loss) / 2

        self.log_dict({"val_g_loss": g_loss, "val_d_loss": d_loss})

    def test_step(self, batch, batch_idx, optimizer_idx):
        return self.validation_step(batch, batch_idx, optimizer_idx)

    def configure_optimizers(self):

        optimizer = getattr(torch.optim, self.optimizer_config.optimizer)
        optimizer_g = optimizer(
            self.generator.parameters(), **self.optimizer_config.optimizer_params
        )
        optimizer_d = optimizer(
            self.discriminator.parameters(), **self.optimizer_config.optimizer_params
        )

        if self.optimizer_config.lr_scheduler is not None:
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, self.optimizer_config.lr_scheduler
            )
            lr_scheduler_g = lr_scheduler(
                optimizer, **self.optimizer_config.lr_scheduler_params
            )
            lr_scheduler_d = lr_scheduler(
                optimizer, **self.optimizer_config.lr_scheduler_params
            )
            return [optimizer_g, optimizer_d], [lr_scheduler_g, lr_scheduler_d]
