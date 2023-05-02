import logging
import os
from logging import StreamHandler
from logging import FileHandler

logger = logging.getLogger("pix2pix")
# 设置为DEBUG级别
logger.setLevel(logging.DEBUG)
# 创建一个格式器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(funcName)s()')


def init_logger(dir):

    # 文件处理器，设置的级别为DEBUG
    file_handler = FileHandler(filename=os.path.join(dir,"debug_log.txt"))
    file_handler.setLevel(logging.NOTSET)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 文件处理器，设置的级别为INFO
    file_handler = FileHandler(filename=os.path.join(dir,"info_log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 文件处理器，设置的级别为WARN
    file_handler = FileHandler(filename=os.path.join(dir,"warn_log.txt"))
    file_handler.setLevel(logging.WARN)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 标准流处理器，设置的级别为INFO
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return get_logger()


def get_logger():
    return logger
