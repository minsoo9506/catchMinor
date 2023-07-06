from catchMinor.base.base_config import loss_func_config, model_config, optimizer_config


class GAN_config(model_config):
    generator_dim_list: list[int] = [2, 4, 8, 16]
    discriminator_dim_list: list[int] = [16, 8, 4, 1]


class GAN_optimizer_config(optimizer_config):
    pass


class GAN_loss_func_config(loss_func_config):
    pass
