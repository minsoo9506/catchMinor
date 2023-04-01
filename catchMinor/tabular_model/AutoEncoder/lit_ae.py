from catchMinor.base.base_lit_model import LitBaseModel
from catchMinor.tabular_model.AutoEncoder.ae_config import (
    AutoEncoder_config,
    AutoEncoder_loss_func_config,
    AutoEncoder_optimizer_config,
)
from catchMinor.tabular_model.AutoEncoder.torch_ae import AutoEncoder
from catchMinor.utils.debug import get_logger


class LitBaseAutoEncoder(LitBaseModel):
    """AutoEncoder with only fully-connected layer lighitning model"""

    def __init__(
        self,
        model_config: AutoEncoder_config,
        optimizer_config: AutoEncoder_optimizer_config,
        loss_func_config: AutoEncoder_loss_func_config,
    ):
        """init LitBaseAutoEncoder

        Args:
            model_config (AutoEncoder_config): config about model
            optimizer_config (AutoEncoder_optimizer_config): config about optimizer
            loss_func_config (AutoEncoder_loss_func_config): config about loss function
        """
        super().__init__(model_config, optimizer_config, loss_func_config)
        logger = get_logger(logger_setLevel="INFO")
        logger.info("AutoEncoder with fully-connected layer is made.")
        self.model = AutoEncoder(model_config)
