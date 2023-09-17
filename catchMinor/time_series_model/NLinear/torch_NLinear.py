# official repo: https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py

import torch
import torch.nn as nn

from catchMinor.time_series_model.NLinear.NLinear_config import NLinear_config


class NLinear(nn.Module):
    """Normalization-Linear model"""

    def __init__(self, config: NLinear_config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # |x| = (batch_size, seq_len, feature_dim)
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        # |x| = (batch_size, pred_len, feature_dim)
        return x
