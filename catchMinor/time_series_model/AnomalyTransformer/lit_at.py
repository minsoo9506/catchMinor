import torch
import torch.nn as nn

from catchMinor.base.base_lit_model import LitBaseModel
from catchMinor.time_series_model.AnomalyTransformer.at_config import (
    AnomalyTransformer_config,
    AnomalyTransformer_loss_func_config,
    AnomalyTransformer_optimizer_config,
)
from catchMinor.time_series_model.AnomalyTransformer.torch_at import AnomalyTransformer
from catchMinor.utils.debug import get_logger


def kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)


class LitAnomalyTransformer(LitBaseModel):
    """AnomalyTransformer lighitning model"""

    def __init__(
        self,
        model_config: AnomalyTransformer_config,
        optimizer_config: AnomalyTransformer_optimizer_config,
        loss_func_config: AnomalyTransformer_loss_func_config,
    ):
        super().__init__(model_config, optimizer_config, loss_func_config)
        self.automatic_optimization = False
        logger = get_logger(logger_setLevel="INFO")
        logger.info("AnomalyTransforme layer is made.")
        self.model = AnomalyTransformer(model_config)
        self.Lambda = model_config.Lambda
        self.n_splits = model_config.n_splits

    def training_step(self, batch, batch_idx):
        x, y = batch
        prior_association_list, series_association_list, y_hat = self.model(x)
        recon_loss = self.loss_func(y_hat, y)
        # calculate Association discrepancy
        prior_loss, series_loss = 0.0, 0.0
        for i in range(len(prior_association_list)):
            prior_loss += torch.mean(
                kl_loss(prior_association_list[i], series_association_list[i].detach())
                + kl_loss(
                    series_association_list[i].detach(), prior_association_list[i]
                )
            )
            series_loss += torch.mean(
                kl_loss(prior_association_list[i].detach(), series_association_list[i])
                + kl_loss(
                    series_association_list[i], prior_association_list[i].detach()
                )
            )
        prior_loss /= len(prior_association_list)
        series_loss /= len(prior_association_list)
        # optimize
        optimizer = self.optimizers()
        # loss
        loss1 = recon_loss - self.Lambda * series_loss
        loss2 = recon_loss + self.Lambda * prior_loss
        # maximize, minimize phase
        optimizer.zero_grad()
        self.manual_backward(loss1, retain_graph=True)
        self.manual_backward(loss2)
        optimizer.step()

        self.log("max_phase_train_loss", loss1, on_epoch=True, prog_bar=True)
        self.log("min_phase_train_loss", loss2, on_epoch=True, prog_bar=True)
        self.log("train_loss", -loss1 + loss2, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prior_association_list, series_association_list, y_hat = self.model(x)
        recon_loss = self.loss_func(y_hat, y)
        # calculate Association discrepancy
        prior_loss, series_loss = 0.0, 0.0
        for i in range(len(prior_association_list)):
            prior_loss += torch.mean(
                kl_loss(prior_association_list[i], series_association_list[i])
                + kl_loss(series_association_list[i], prior_association_list[i])
            )
            series_loss += torch.mean(
                kl_loss(prior_association_list[i], series_association_list[i])
                + kl_loss(series_association_list[i], prior_association_list[i])
            )
        prior_loss /= len(prior_association_list)
        series_loss /= len(prior_association_list)
        # maximize phase
        # forces the series-association to pay more attention to the non-adjacent horizon
        # then, hard to reconstruct abnormal point
        loss1 = recon_loss - self.Lambda * series_loss
        # minimize phase
        loss2 = recon_loss + self.Lambda * prior_loss

        self.log("max_phase_valid_loss", loss1, on_epoch=True, prog_bar=True)
        self.log("min_phase_valid_loss", loss2, on_epoch=True, prog_bar=True)
        self.log("valid_loss", -loss1 + loss2, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    def get_anomaly_score(self, batch, data_type: str = "time_series") -> torch.Tensor:
        """get anomaly score in each points

        Args:
            batch (torch.Tensor): |batch| = (batch_size, input_length(window_size), feature_dim), non overlaps window
            data_type (str): fixed as 'time_series'

        Returns:
            anomaly score (torch.Tensor): anomaly score of each points
        """
        x, y = batch
        # |x| = (batch_size, input_length(window_size), feature_dim)
        prior_association_list, series_association_list, y_hat = self.model(x)
        return_shape = x.size(0) * x.size(1)
        association_loss = torch.zeros((return_shape,))
        for i in range(len(prior_association_list)):
            prior_association = prior_association_list[i].split(x.size(0), dim=0)
            series_association = series_association_list[i].split(x.size(0), dim=0)
            for j in range(self.n_splits):
                association_loss += (
                    kl_loss(prior_association[j], series_association[j])
                    + kl_loss(series_association[j], prior_association[j])
                ).reshape(return_shape)
        association_loss /= len(prior_association_list)
        recon_loss = torch.mean((y - y_hat) ** 2, dim=-1).reshape(return_shape)
        anomaly_score = nn.functional.softmax(-association_loss) * recon_loss
        # |anomaly_score| = (batch_size * input_length, )
        return anomaly_score
