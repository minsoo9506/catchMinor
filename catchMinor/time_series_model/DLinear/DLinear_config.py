from typing import Optional

from catchMinor.base.base_config import loss_func_config, model_config, optimizer_config


class DLinear_config(model_config):
    features_dim_list: list[int] = [16, 8, 4, 2]
    kernel_size: int = 25
    seq_len: int = 128


class DLinear_optimizer_config(optimizer_config):
    optimizer: str = "Adam"
    optimizer_params: Optional[dict] = {"lr": 0.0001}


class DLinear_loss_func_config(loss_func_config):
    loss_fn: str = "MSELoss"
