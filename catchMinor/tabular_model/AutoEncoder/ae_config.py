from catchMinor.base.base_config import loss_func_config, model_config, optimizer_config


class AutoEncoder_config(model_config):
    features_dim_list: list[int] = [16, 8, 4, 2]


class AutoEncoder_optimizer_config(optimizer_config):
    pass


class AutoEncoder_loss_func_config(loss_func_config):
    pass
