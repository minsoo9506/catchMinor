import torch
import torch.nn.functional as F

from catchMinor.base.base_lit_model import LitBaseModel
from catchMinor.tabular_model.VAE.torch_vae import VAE
from catchMinor.tabular_model.VAE.vae_config import (
    VAE_config,
    VAE_loss_func_config,
    VAE_optimizer_config,
)
from catchMinor.utils.debug import get_logger


def loss_function(output_x, x, mu, logvar):
    BCE = F.mse_loss(output_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


class LitVAE(LitBaseModel):
    """VAE with only fully-connected layer lighitning model"""

    def __init__(
        self,
        model_config: VAE_config,
        optimizer_config: VAE_loss_func_config,
        loss_func_config: VAE_optimizer_config,
    ):

        super().__init__(model_config, optimizer_config, loss_func_config)
        logger = get_logger(logger_setLevel="INFO")
        logger.info("VAE with fully-connected layer is made.")
        self.model = VAE(model_config)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        output_x, mu, logvar = self(x)
        loss = loss_function(output_x, x, mu, logvar)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        output_x, mu, logvar = self(x)
        loss = loss_function(output_x, x, mu, logvar)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        output_x, mu, logvar = self(x)
        loss = loss_function(output_x, x, mu, logvar)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def get_anomaly_score(self, batch) -> torch.Tensor:
        """get anomaly score

        Args:
            batch (torch.Tensor): _description_

        Returns:
            anomaly score (torch.Tensor): mean(abs(true - pred), dim=1)
        """
        x, y = batch
        output, _, _ = self.model(x)
        anomaly_score = torch.abs(y - output)
        return torch.mean(anomaly_score, dim=1)

    def _configure_loss_func(self, loss_func_config: VAE_loss_func_config):
        pass
