from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from catchMinor.base.base_torch_model import TorchBaseModel
from catchMinor.tabular_model.VAE.vae_config import VAE_config


class Encoder(TorchBaseModel):
    """encoder in VAE"""

    def __init__(self, config: VAE_config) -> None:
        super().__init__(config)

    def _build_layers(self, config: VAE_config) -> torch.nn.Sequential:
        layers: list[Any] = []
        for in_features, out_features in zip(
            config.features_dim_list[:-2], config.features_dim_list[1:-1]
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
    """decoder in VAE"""

    def __init__(self, config: VAE_config) -> None:
        super().__init__(config)

    def _build_layers(self, config: VAE_config) -> torch.nn.Sequential:
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


class VAE(nn.Module):
    """VAE"""

    def __init__(self, config: VAE_config) -> None:
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.encoder_mu_layer = nn.Linear(
            config.features_dim_list[-2], config.features_dim_list[-1]
        )
        self.encoder_logvar_layer = nn.Linear(
            config.features_dim_list[-2], config.features_dim_list[-1]
        )

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """encode part in vae (get distribution of latent z)

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            mu, logvar
        """
        h = self.encoder(x)
        mu = self.encoder_mu_layer(h)
        logvar = self.encoder_logvar_layer(h)
        return mu, logvar

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """reparameterize in vae

        Parameters
        ----------
        mu : torch.Tensor
            _description_
        logvar : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            reparameterized result of encoder (= mu + esp * std)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        # rand_like
        # Returns a Tensor with the same size as input
        # that is filled with random numbers from a uniform distribution
        return mu + eps * std

    def _decode(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """decode part in vae

        Parameters
        ----------
        z : torch.Tensor
            latent variable

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            z, mu, logvar
        """

        x = self.decoder(z)
        x = torch.sigmoid(x)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            reconstructed x, mu, logvar
        """
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        return self._decode(z), mu, logvar
