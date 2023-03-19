from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class tabularDataset(Dataset):
    """custom torch dataset for tabular data"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Args:
            x (np.ndarray): features
            y (np.ndarray): targets(labels)
        """
        super().__init__()

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx, :], self.y[idx, :]


# TODO: univariate랑 multivariate 구분없이 사용할 수 있도록 아래 코드 수정
# univariate일 때, 그냥 dummy axis 하나 더 있는거지!


class UnivariateBaseWindowDataset(Dataset):
    """Window based dataset in univariate time series data"""

    def __init__(self, x: np.ndarray, y: np.ndarray, window_size: int):
        """
        Args:
            x (np.ndarray): input feature
            y (np.ndarray): input label
            window_size (int): window size of input
        """

        super().__init__()

        data_len = len(x) - window_size + 1
        self.x = np.zeros((data_len, window_size))
        self.y = np.zeros((data_len, window_size))

        for idx in range(data_len):
            self.x[idx, :] = x[idx : idx + window_size]
            self.y[idx, :] = y[idx : idx + window_size]

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        return self.x[idx, :], self.y[idx, :]


# LSTM input = (N,L,H) when batch_first=True
# = (batch_size, seq_length, hidden_size)
class UnivariateLSTMWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, window_size: int):
        super().__init__()

        data_len = len(x) - window_size + 1
        self.x = np.zeros((data_len, window_size))
        self.y = np.zeros((data_len, window_size))

        for idx in range(data_len):
            self.x[idx, :] = x[idx : idx + window_size]
            self.y[idx, :] = y[idx : idx + window_size]

        # add axis (hidden_size)
        self.x = self.x[:, :, np.newaxis]
        self.y = self.y[:, :, np.newaxis]

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        return self.x[idx, :], self.y[idx, :]


if __name__ == "__main__":
    # univariate time series data -> window-based approach
    dataset = UnivariateLSTMWindowDataset(
        x=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        y=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        window_size=3,
    )
    data_loader = DataLoader(dataset, batch_size=2)
    for x, _ in data_loader:
        print(x.shape)
        print(x)
        break
