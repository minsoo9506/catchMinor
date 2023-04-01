from abc import ABCMeta, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from catchMinor.base.base_config import model_config


class TorchBaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, config: model_config):
        super().__init__()
        self.config = config
        self.activation_func = self._configure_activation_func(
            config.activation_func_name
        )
        self.model = self._build_layers(config)

    @abstractmethod
    def _build_layers(self, config: model_config) -> Any:
        pass

    def _configure_activation_func(self, activation_func_name: str) -> torch.nn:
        activation_func = getattr(nn, activation_func_name)
        return activation_func()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out
