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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx, :], self.y[idx, :]


class WindowDataset(Dataset):
    """Window-based dataset in univariate & multivariate time series data.

    if x.dim is 1, then feature_dim will be 1.
    |self.x| = (data_len, window_size, feature_dim)
    |self.y| = (data_len, window_size, feature_dim)
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window_size: int,
        overlaps: bool,
        shape: str = "wf",
    ):
        """
        Args:
            x (np.ndarray): input feature
            y (np.ndarray): input label
            window_size (int): window size of input
            overlaps (bool) : true, if want to make overlap dataset
            shape (str) : wf is (window_size, feature_dim), fw is (feature_dim, window_size)
        """
        super().__init__()

        assert shape in ["wf", "fw"], "shape has to be wf or fw"

        self.x = self._make_window_based_data(x, window_size, overlaps, shape)
        self.y = self._make_window_based_data(y, window_size, overlaps, shape)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        return self.x[idx, :], self.y[idx, :]

    def _make_window_based_data(
        self, data: np.ndarray, window_size: int, overlaps: int, shape: str
    ) -> np.ndarray:
        if data.ndim == 1:
            data = np.expand_dims(data, axis=-1)
        if overlaps:
            _data = self._overlaps(data, window_size, shape)
        else:
            _data = self._non_overlaps(data, window_size, shape)
        return _data

    def _overlaps(self, data: np.ndarray, window_size: int, shape: str) -> np.ndarray:
        data_len = len(data) - window_size + 1
        feature_dim = data.shape[-1]
        if shape == "wf":
            _data = np.zeros((data_len, window_size, feature_dim))
            for idx in range(data_len):
                _data[idx, :] = data[idx : idx + window_size]
        else:
            _data = np.zeros((data_len, feature_dim, window_size))
            for idx in range(data_len):
                _data[idx, :] = data[idx : idx + window_size].T
        return _data

    def _non_overlaps(
        self, data: np.ndarray, window_size: int, shape: str
    ) -> np.ndarray:
        data_len = len(data) // window_size
        feature_dim = data.shape[-1]
        if shape == "wf":
            _data = np.zeros((data_len, window_size, feature_dim))
            for idx in range(data_len):
                _data[idx, :] = data[idx * window_size : (idx + 1) * window_size]
        else:
            _data = np.zeros((data_len, feature_dim, window_size))
            for idx in range(data_len):
                _data[idx, :] = data[idx * window_size : (idx + 1) * window_size].T
        return _data


# LSTM input = (N,L,H) when batch_first=True
# = (batch_size, seq_length, hidden_size)
# class UnivariateLSTMWindowDataset(Dataset):
#     def __init__(self, x: np.ndarray, y: np.ndarray, window_size: int):
#         super().__init__()

#         data_len = len(x) - window_size + 1
#         self.x = np.zeros((data_len, window_size))
#         self.y = np.zeros((data_len, window_size))

#         for idx in range(data_len):
#             self.x[idx, :] = x[idx : idx + window_size]
#             self.y[idx, :] = y[idx : idx + window_size]

#         # add axis (hidden_size)
#         self.x = self.x[:, :, np.newaxis]
#         self.y = self.y[:, :, np.newaxis]

#         self.x = torch.tensor(self.x, dtype=torch.float32)
#         self.y = torch.tensor(self.y, dtype=torch.float32)

#     def __len__(self) -> int:
#         return self.x.shape[0]

#     def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
#         return self.x[idx, :], self.y[idx, :]


if __name__ == "__main__":
    # univariate time series data
    uni_dataset = WindowDataset(
        x=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        y=np.array([1, 0, 0, 0, 1, 0, 0, 1, 0]),
        window_size=2,
        overlaps=True,
        shape="wf",
    )
    data_loader = DataLoader(uni_dataset, batch_size=2)
    print(">> univariate time series dataset")
    for x, _ in data_loader:
        print(x.shape)
        print(x)
        break

    # multivariate time series data
    multi_dataset = WindowDataset(
        x=np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]),
        y=np.array([1, 0, 0, 0, 1, 0, 0, 1, 0]),
        window_size=2,
        overlaps=True,
        shape="wf",
    )
    data_loader = DataLoader(multi_dataset, batch_size=2)
    print(">> multivariate time series dataset")
    for x, _ in data_loader:
        print(x.shape)
        print(x)
        break
