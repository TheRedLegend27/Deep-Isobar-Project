from loguru import logger


def get_logger(name: str):
    return logger.bind(module=name)