from catchMinor.base.base_lit_model import LitBaseModel
from catchMinor.time_series_model.DLinear.DLinear_config import (
    DLinear_config,
    DLinear_loss_func_config,
    DLinear_optimizer_config,
)
from catchMinor.time_series_model.DLinear.torch_DLinear import DLinear
from catchMinor.utils.debug import get_logger


class LitDLinear(LitBaseModel):
    """DLinear lighitning model"""

    def __init__(
        self,
        model_config: DLinear_config,
        optimizer_config: DLinear_optimizer_config,
        loss_func_config: DLinear_loss_func_config,
    ):
        super().__init__(model_config, optimizer_config, loss_func_config)
        logger = get_logger(logger_setLevel="INFO")
        logger.info("DLinear layer is made.")
        self.model = DLinear(model_config)
