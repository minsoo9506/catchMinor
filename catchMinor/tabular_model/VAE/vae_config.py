from catchMinor.base.base_config import loss_func_config, model_config, optimizer_config


class VAE_config(model_config):
    features_dim_list: list[int] = [16, 8, 4, 2]


class VAE_optimizer_config(optimizer_config):
    pass


class VAE_loss_func_config(loss_func_config):
    pass
