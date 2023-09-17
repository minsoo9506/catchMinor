from typing import Optional

from catchMinor.base.base_config import loss_func_config, model_config, optimizer_config


class NLinear_config(model_config):
    seq_len: int
    pred_len: int


class NLinear_optimizer_config(optimizer_config):
    optimizer: str = "Adam"
    optimizer_params: Optional[dict] = {"lr": 0.0001}


class NLinear_loss_func_config(loss_func_config):
    loss_fn: str = "MSELoss"
