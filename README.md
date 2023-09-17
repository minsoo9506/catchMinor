# catchMinor
- all about imbalanced-learning, anomaly-detection (tabular, time series, graph)
- focus on developing models based on torch, lightning

## Example Code
```python
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from catchMinor.data_load.dataset import tabularDataset
from catchMinor.tabular_model.AutoEncoder.ae_config import (
    AutoEncoder_config,
    AutoEncoder_loss_func_config,
    AutoEncoder_optimizer_config,
)
from catchMinor.tabular_model.AutoEncoder.lit_ae import LitBaseAutoEncoder
from catchMinor.utils.data import normal_only_train_split_tabular

# set data
data = np.load(data_path)
X, y = data["X"], data["y"]

(
    normal_X_train,
    mix_X_test,
    normal_y_train,
    mix_y_test,
) = normal_only_train_split_tabular(X, y, 0.8)

train_dataset = tabularDataset(normal_X_train, deepcopy(normal_X_train))
valid_dataset = tabularDataset(mix_X_test, deepcopy(mix_X_test))

train_loader = DataLoader(train_dataset, batch_size=512)
valid_loader = DataLoader(valid_dataset, batch_size=512)

# config
model_config = AutoEncoder_config(features_dim_list=[9, 4])
optim_config = AutoEncoder_optimizer_config()
loss_func_config = AutoEncoder_loss_func_config(loss_fn="MSELoss")

# Lit model
model = LitBaseAutoEncoder(model_config, optim_config, loss_func_config)

# trainer
TensorBoard_logger = TensorBoardLogger(
    save_dir="./log", name="AutoEncoder", version="0.1"
)

early_stopping_callback = EarlyStopping(monitor="valid_loss", mode="min", patience=2)

trainer = Trainer(
    log_every_n_steps=1,
    accelerator=config.cuda,
    logger=TensorBoard_logger,
    max_epochs=config.epochs,
    deterministic=True,
    callbacks=[early_stopping_callback],
    check_val_every_n_epoch=1,
)

# fit the model
trainer.fit(model, train_loader, valid_loader)
```


## Installation (not yet)
- recommend to use `pyenv` virtualenv
```bash
python -m pip install catchMinor
```

## Benchmark
- tabluar data: [ADBench](https://github.com/Minqi824/ADBench)
    - (paper) [ADBench: Anomaly Detection Benchmark](https://arxiv.org/abs/2206.09426)
- time series data: UCR Time Series Anomaly Archive
    - (paper) [Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress](https://arxiv.org/abs/2009.13807)

## Implemented Algorithms
### Anomaly Detection
|model|data|desc|
|:---:|:---:|:---:|
|AutoEncoder|tabular, time series|linear, reconstruction-based|
|VAE|tabular, time series|linear, reconstruction-based|
|GAN|tabular, time series|linear, reconstruction-based|
|AnomalyTransformer|time series|transformer|
|DLienar|time series|linear, pred-based|
|NLienar|time series|linear, pred-based|

### Imbalanced Learning

## Contribute
- follow gitflow & forkflow
- branch
    - `master`: production branch
    - `develop`: develop branch
    - `feature/{work}`: feature branch
    - `release/{version}`: temporary branch before release
    - `hotfix`: bugfix branch based on master branch
- merge (main contributor)
    - merge from `feature` branch to `develop` branch: `merge squash`
    - else: `merge --no-ff`
- example
```bash
# 1. fork repository
# 2. clone your origin and add remote upstream
git clone {your_origin_repo}
git add remote upstream {catchMinor_origin_repo}
# 3. make your origin develop branch as default branch
# 4. make feature branch
git checkout -b feature/{your_work} develop
# 5. do something to contribute
# 6. add, commit (use cz c)
git add {your_work}
git commit -m "{your_work_message}"
# 7. push to your origin (develop branch)
git push origin feature/{your_work}
# 8. PR in github
```