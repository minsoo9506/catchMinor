from typing import Optional

from catchMinor.base.base_config import loss_func_config, model_config, optimizer_config


class AnomalyTransformer_config(model_config):
    input_length: int = 128
    feature_dim: int = 1
    hidden_size: int = 512
    n_splits: int = 8
    n_enc_blocks: int = 3
    dropout_p: float = 0.1
    use_leaky_relu: bool = False
    Lambda: float = 3.0


class AnomalyTransformer_optimizer_config(optimizer_config):
    optimizer: str = "Adam"
    optimizer_params: Optional[dict] = {"lr": 0.0001}


class AnomalyTransformer_loss_func_config(loss_func_config):
    loss_fn: str = "MSELoss"
