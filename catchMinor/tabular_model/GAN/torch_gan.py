from typing import Any

import torch
import torch.nn as nn

from catchMinor.base.base_torch_model import TorchBaseModel
from catchMinor.tabular_model.GAN.gan_config import GAN_config


class Generator(TorchBaseModel):
    def __init__(self, config: GAN_config):
        super().__init__(config)

    def _build_layers(self, config: GAN_config) -> torch.nn.Sequential:
        layers: list[Any] = []
        for in_features, out_features in zip(
            config.generator_dim_list[:-2], config.generator_dim_list[1:-1]
        ):
            layers.append(nn.Linear(in_features, out_features))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(self.activation_func)
            if config.dropout_p != 0:
                layers.append(nn.Dropout(config.dropout_p))
            layers.append(
                nn.Linear(
                    in_features=config.generator_dim_list[-2],
                    out_features=config.generator_dim_list[-1],
                )
            )
            layers.append(nn.Tanh())
        model = nn.Sequential(*layers)
        return model


class Discriminator(TorchBaseModel):
    def __init__(self, config: GAN_config):
        super().__init__(config)

    def _build_layers(self, config: GAN_config) -> torch.nn.Sequential:
        layers: list[Any] = []
        for in_features, out_features in zip(
            config.discriminator_dim_list[:-2], config.discriminator_dim_list[1:-1]
        ):
            layers.append(nn.Linear(in_features, out_features))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(self.activation_func)
            if config.dropout_p != 0:
                layers.append(nn.Dropout(config.dropout_p))
            layers.append(
                nn.Linear(
                    in_features=config.discriminator_dim_list[-2],
                    out_features=config.discriminator_dim_list[-1],
                )
            )
            layers.append(nn.Sigmoid())
        model = nn.Sequential(*layers)
        return model
