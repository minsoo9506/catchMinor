import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from catchMinor.data_load.dataset import WindowDataset
from catchMinor.time_series_model.AnomalyTransformer.at_config import (
    AnomalyTransformer_config,
    AnomalyTransformer_loss_func_config,
    AnomalyTransformer_optimizer_config,
)
from catchMinor.time_series_model.AnomalyTransformer.lit_at import LitAnomalyTransformer


def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", default="Time Series Anomaly Detection")
    parser.add_argument("--model", default="LitAnomalyTransformer")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="input batch size for training (default: 512)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 2)"
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


if __name__ == "__main__":
    config = define_argparser()
    seed_everything(0)

    msl_train = np.load("../data/timeseries/AnomalyTransformer/MSL/MSL_train.npy")
    msl_test_label = np.load(
        "../data/timeseries/AnomalyTransformer/MSL/MSL_test_label.npy"
    )
    msl_test = np.load("../data/timeseries/AnomalyTransformer/MSL/MSL_test.npy")

    anomaly_ratio = msl_test_label.sum() / len(msl_test_label)
    print(f"Anomaly Ratio in test dataset is {anomaly_ratio * 100:.2f} %")

    scaler = StandardScaler()
    scaler.fit(msl_train)
    scaled_train = scaler.transform(msl_train)
    scaled_test = scaler.transform(msl_test)
    train_ratio = 0.8
    num_train = int(len(scaled_train) * train_ratio)

    train = scaled_train[:num_train, :]
    valid = scaled_train[num_train:, :]

    print(f"train.shape={train.shape}")
    print(f"valid.shape={valid.shape}")
    print(f"test.shape={scaled_test.shape}")

    train_dataset = WindowDataset(
        train, train, window_size=128, overlaps=False, shape="wf"
    )
    valid_dataset = WindowDataset(
        valid, valid, window_size=128, overlaps=False, shape="wf"
    )
    test_dataset = WindowDataset(
        scaled_test, scaled_test, window_size=128, overlaps=False, shape="wf"
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # config
    model_config = AnomalyTransformer_config(feature_dim=55)
    loss_config = AnomalyTransformer_loss_func_config()
    optim_config = AnomalyTransformer_optimizer_config()

    model = LitAnomalyTransformer(model_config, optim_config, loss_config)

    # callback: tensorboard
    TensorBoard_logger = TensorBoardLogger(
        save_dir="./log", name=config.model, version=config.current_time
    )

    # callback: progrss bar
    rich_progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="Anomaly Detection",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        ),
        leave=True,
    )

    # callback: save the best model in every epochs
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath="./checkpoints/",
        filename="model_name-{epoch}-{valid_acc:.4f}",
        save_top_k=1,
        mode="min",
    )

    # callback: early stop
    early_stopping_callback = EarlyStopping(
        monitor="valid_loss", mode="min", patience=2
    )

    # trainer
    trainer = Trainer(
        log_every_n_steps=1,
        accelerator=config.cuda,
        logger=TensorBoard_logger,
        max_epochs=config.epochs,
        deterministic=True,
        callbacks=[early_stopping_callback, rich_progress_bar, checkpoint_callback],
        check_val_every_n_epoch=1,
    )

    # fit the model
    trainer.fit(model, train_loader, valid_loader)

    # load best model (min train_loss)
    checkpoint = torch.load(checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint["state_dict"])

    # calculate anomaly score
    with torch.no_grad():
        model.eval()
        anomaly_scores = []
        for batch in test_loader:
            anomaly_score = model.get_anomaly_score(batch).detach().numpy().tolist()
            anomaly_scores += anomaly_score

    short = min(len(msl_test_label), len(anomaly_scores))
    result = pd.DataFrame(
        {"label": msl_test_label[:short], "anomaly_score": anomaly_scores[:short]}
    )
    print(result.groupby("label")["anomaly_score"].mean())
