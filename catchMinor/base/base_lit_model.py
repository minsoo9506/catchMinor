import pytorch_lightning as pl
import torch
import torch.optim

from catchMinor.base.base_config import loss_func_config, model_config, optimizer_config


class LitBaseModel(pl.LightningModule):
    def __init__(
        self,
        model_config: model_config,
        optimizer_config: optimizer_config,
        loss_func_config: loss_func_config,
    ):

        super().__init__()
        # self.model = None
        self.loss_func = self._configure_loss_func(loss_func_config)
        self.optimizer_config = optimizer_config

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_config.optimizer)
        optimizer = optimizer(
            self.model.parameters(), **self.optimizer_config.optimizer_params
        )

        if self.optimizer_config.lr_scheduler is not None:
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, self.optimizer_config.lr_scheduler
            )
            lr_scheduler = lr_scheduler(
                optimizer, **self.optimizer_config.lr_scheduler_params
            )
            return [optimizer], [lr_scheduler]

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.loss_func(output, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.loss_func(output, y)
        self.log("val_loss", loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.loss_func(output, y)
        self.log("test_loss", loss, on_epoch=True)

    def _configure_loss_func(self, loss_func_config: loss_func_config):
        loss_func = getattr(torch.nn, loss_func_config.loss_fn)
        loss_func = loss_func(**loss_func_config.loss_fn_params)
        return loss_func
