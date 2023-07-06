import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

msl_train = np.load('../data/timeseries/AnomalyTransformer/MSL/MSL_train.npy')
msl_test_label = np.load('../data/timeseries/AnomalyTransformer/MSL/MSL_test_label.npy')
msl_test = np.load('../data/timeseries/AnomalyTransformer/MSL/MSL_test.npy')


anomaly_ratio = msl_test_label.sum() / len(msl_test_label)
print(f'Anomaly Ratio in test dataset is {anomaly_ratio * 100:.2f} %')

import argparse
from copy import deepcopy
from datetime import datetime

import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger

def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", default="Time Series Anomaly Detection")
    parser.add_argument("--model", default="LitAnomalyTransformer")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="input batch size for training (default: 512)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of epochs to train (default: 2)"
    )
    parser.add_argument("--cuda", type=int, default=0, help="0 for cpu -1 for all gpu")

    config = parser.parse_args()  # jupyter에서는 args=[] 이용

    if config.cuda == 0 or not torch.cuda.is_available():
        config.cuda = "cpu"
    else:
        config.cuda = "gpu"

    current_time = str(datetime.today()).split(".")[0]
    config.current_time = current_time

    return config


config = define_argparser()

from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader

scaler = StandardScaler()
scaler.fit(msl_train)
scaled_train = scaler.transform(msl_train)

train_ratio = 0.8
num_train = int(len(scaled_train)*train_ratio)

train = scaled_train[:num_train, :]
valid = scaled_train[num_train:, :]

print(f'train.shape={train.shape}')
print(f'valid.shape={valid.shape}')

from catchMinor.data_load.dataset import WindowDataset

train_dataset = WindowDataset(train, train, window_size=128, overlaps=False, shape='fw')
valid_dataset = WindowDataset(valid, valid, window_size=128, overlaps=False, shape='fw')

train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)

from catchMinor.time_series_model.AnomalyTransformer.at_config import *

model_config = AnomalyTransformer_config(feature_dim=55)
loss_config = AnomalyTransformer_loss_func_config()
optim_config = AnomalyTransformer_optimizer_config()