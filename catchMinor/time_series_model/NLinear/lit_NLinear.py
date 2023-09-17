from catchMinor.base.base_lit_model import LitBaseModel
from catchMinor.time_series_model.NLinear.NLinear_config import (
    NLinear_config,
    NLinear_loss_func_config,
    NLinear_optimizer_config,
)
from catchMinor.time_series_model.NLinear.torch_NLinear import NLinear
from catchMinor.utils.debug import get_logger


class LitNLinear(LitBaseModel):
    """NLinear lighitning model"""

    def __init__(
        self,
        model_config: NLinear_config,
        optimizer_config: NLinear_loss_func_config,
        loss_func_config: NLinear_optimizer_config,
    ):
        super().__init__(model_config, optimizer_config, loss_func_config)
        logger = get_logger(logger_setLevel="INFO")
        logger.info("NLinear layer is made.")
        self.model = NLinear(model_config)
