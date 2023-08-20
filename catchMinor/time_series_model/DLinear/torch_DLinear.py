# decompose by moving average kernel
# raw = trend component + seasonal component
# official repo: https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py

import torch
import torch.nn as nn

from catchMinor.time_series_model.DLinear.DLinear_config import DLinear_config


class MovingAverage(nn.Module):
    """
    Moving average block
    """

    def __init__(self, kernel_size: int):
        super(MovingAverage, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size has to be odd number"
        self.kernel_size = kernel_size
        self.moving_average_layer = nn.AvgPool1d(
            kernel_size=self.kernel_size, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # |x| = (batch_size, length, feature_dim)
        # padding on the both ends of time series for |input| = |output after ma|
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.moving_average_layer(
            x.permute(0, 2, 1)
        )  # permute: nn.AvgPool1d's input shape has to be (batch_size, feature_dim, length)
        x = x.permute(0, 2, 1)
        # |x| = (batch_size, length, feature_dim)
        return x


class SeriesDecomposition(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size: int):
        super(SeriesDecomposition, self).__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moving_average_result = self.moving_average(x)
        res = x - moving_average_result
        return res, moving_average_result


class DLinear(nn.Module):
    """Decompotision Linear model"""

    def __init__(self, config: DLinear_config):
        super(DLinear, self).__init__()
        self.decompsition = SeriesDecomposition(config.kernel_size)

        self.Linear_Seasonal = nn.Linear(config.seq_len, config.seq_len)
        self.Linear_Trend = nn.Linear(config.seq_len, config.seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # |x| = (batch_size, seq_len, feature_dim)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        # |x| = (batch_size, feature_dim, seq_len)
        return x.permute(0, 2, 1)  # |x| = (batch_size, seq_len, feature_dim)
