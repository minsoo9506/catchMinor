from datetime import datetime
from typing import Dict, Optional

import torch
from pydantic import BaseModel, Field, validator


class model_config(BaseModel):
    activation_func_name: str = "ReLU"
    dropout_p: Optional[float] = 0.2
    use_batch_norm: Optional[bool] = False  # normalizer는 따로...?
    generated_time: datetime = Field(
        default_factory=datetime.now, description="time when instance is generated"
    )

    @validator("activation_func_name")
    def valid_activation_func_name(cls, v: str):
        try:
            _ = getattr(torch.nn, v)
        except AttributeError:
            print(f"{v} is not a valid activation function defined in the torch.nn")
        return v


class optimizer_config(BaseModel):
    optimizer: str = "Adam"
    optimizer_params: Optional[Dict] = {}
    lr_scheduler: Optional[str] = None
    lr_scheduler_params: Optional[Dict] = {}
    generated_time: datetime = Field(
        default_factory=datetime.now, description="time when instance is generated"
    )

    @validator("optimizer")
    def valid_optimizer(cls, v: str):
        try:
            _ = getattr(torch.optim, v)
        except AttributeError:
            print(f"{v} is not a valid optimizer defined in the torch.optim")
        return v

    @validator("lr_scheduler")
    def valid_scheduler(cls, v: str):
        try:
            _ = getattr(torch.optim.lr_scheduler, v)
        except AttributeError:
            print(
                f"{v} is not a valid optimizer defined in the torch.optim.lr_scheduler"
            )
        return v


class loss_func_config(BaseModel):
    loss_fn: str
    loss_fn_params: Optional[Dict] = {}
    generated_time: datetime = Field(
        default_factory=datetime.now, description="time when instance is generated"
    )

    @validator("loss_fn")
    def valid_loss_fn(cls, v: str):
        try:
            _ = getattr(torch.nn, v)
        except AttributeError:
            print(f"{v} is not a valid loss function defined in the torch.nn")
        return v
