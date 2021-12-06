import logging
import os
import pickle
import sys
from typing import Callable, Any


_DNN_ABR = {'alexnet': 'ax', 'vgg16': 'vg16', 'googlenet': 'gn', 'resnet50': 'rs50'}


def dnn_abbr(dnn_loader: Callable) -> str:
    """对DNN名称的缩写"""
    return _DNN_ABR[dnn_loader.__name__.replace('prepare_', '')]


def cached_func(file_name: str, func: Callable, *args,
                prefix: str = '.cache', logger: logging.Logger = None) -> Any:
    """对于执行耗时较长的函数，将其运行结果用pickle序列化，缓存在.cache目录下
    :param file_name 缓存文件名称，通过检查该文件名是否存在确定是否执行func
    :param func 要执行的函数
    :param args 函数的参数
    :param prefix 缓存文件的相对路径，默认为当前执行路径下的.cache目录
    :param logger 写入的logger，默认logger名为func的函数名，写入stdout
    :return 函数的结果
    """
    if logger is None:
        # 如果没有传入logger，则默认写入stdout
        logger = logging.getLogger(func.__name__)
        if not logger.hasHandlers():
            # 因为这个logger是全局共用的，所以不能重复添加Handler
            logger.addHandler(logging.StreamHandler(sys.stdout))
            logger.setLevel(logging.DEBUG)
    file_path = prefix + '/' + file_name
    if os.path.isfile(file_path):
        logger.debug(f"{file_name} exists, loading...")
        with open(file_path, 'rb') as cfile:
            return pickle.load(cfile)
    else:
        if not os.path.isdir('.cache'):
            os.mkdir('.cache')
        logger.debug(f"{file_name} not exists, generating...")
        data = func(*args)
        logger.debug(f"{file_name} generated, writing...")
        with open(file_path, 'wb') as cfile:
            pickle.dump(data, cfile)
        return data
