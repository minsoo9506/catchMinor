from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from catchMinor.base.base_torch_model import TorchBaseModel
from catchMinor.tabular_model.AutoEncoder.ae_config import AutoEncoder_config


class Encoder(TorchBaseModel):
    """Encoder of AutoEncoder"""

    def __init__(self, config: AutoEncoder_config):
        """init Encoder

        Args:
            config (AutoEncoder_config): AutoEncoder's config
        """
        super().__init__(config)

    def _build_layers(self, config: AutoEncoder_config) -> torch.nn.Sequential:
        layers: list[Any] = []
        for in_features, out_features in zip(
            config.features_dim_list[:-1], config.features_dim_list[1:]
        ):
            layers.append(nn.Linear(in_features, out_features))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(self.activation_func)
            if config.dropout_p != 0:
                layers.append(nn.Dropout(config.dropout_p))
        model = nn.Sequential(*layers)
        return model


class Decoder(TorchBaseModel):
    """Decoder of AutoEncoder"""

    def __init__(self, config: AutoEncoder_config):
        """init Decoder

        Args:
            config (AutoEncoder_config): AutoEncoder's config
        """
        super().__init__(config)

    def _build_layers(self, config: AutoEncoder_config) -> torch.nn.Sequential:
        reversed_features_dim_list = deepcopy(config.features_dim_list)
        reversed_features_dim_list.reverse()
        layers: list[Any] = []
        for in_features, out_features in zip(
            reversed_features_dim_list[:-1], reversed_features_dim_list[1:]
        ):
            layers.append(nn.Linear(in_features, out_features))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(self.activation_func)
            if config.dropout_p != 0:
                layers.append(nn.Dropout(config.dropout_p))

        model = nn.Sequential(*layers)
        return model


class AutoEncoder(nn.Module):
    """AutoEncoder torch model"""

    def __init__(self, config: AutoEncoder_config):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x
