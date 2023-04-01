import logging


def get_logger(
    logger_name: str = __name__, logger_setLevel: str = "INFO"
) -> logging.Logger:
    """make logger and return it

    Args:
        logger_name (str, optional): logger name. Defaults to __name__.
        logger_setLevel (str, optional): argument in `logger.setLevel`. Defaults to "INFO".

    Returns:
        logging.Logger: made logger
    """
    logger = logging.getLogger(logger_name)
    level = getattr(logging, logger_setLevel, "INFO")
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


if __name__ == "__main__":
    logger = get_logger(__name__, "INFO")
    logger.debug("debug")
    logger.info("info")
