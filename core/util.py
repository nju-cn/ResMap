import logging
import os
import pickle
import sys
import time
from enum import Enum, auto
from typing import Callable, Any, Type

from google.protobuf.message import Message

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


class Timer:
    def __init__(self):
        self._begin = 0
        self._cost = 0

    def __enter__(self):
        self._begin = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cost = time.time() - self._begin

    def cost(self):
        return self._cost


class ActTimer(Timer):
    def __init__(self, act_name: str, logger: logging.Logger):
        super().__init__()
        self._act_name = act_name
        self._logger = logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self._logger.debug(f"{self._act_name} costs {round(self.cost(), 2)}s")


class SerialTimer(ActTimer):
    class SType(Enum):
        DUMP = auto()
        LOAD = auto()

    def __init__(self, s_type: SType, obj_type: Type, logger: logging.Logger):
        act_name = ('Dumping' if s_type == self.SType.DUMP else 'Loading')
        super().__init__(f"{act_name} {obj_type.__name__}", logger)


def timed_rpc(rpc_func: Callable, req_msg: Message, dest: str, mode: str, logger: logging.Logger) -> Message:
    """对整个rpc计时，rpc_func应该只有发送或接收明显耗时，mode为s表示发送，r表示接收"""
    with Timer() as timer:
        rsp_msg = rpc_func(req_msg)
    if 's' in mode:
        mb_size = req_msg.ByteSize() / 1024 / 1024  # 单位MB
        # 计算网速时，添加极小量避免本地模拟时出现除零异常
        logger.debug(f"Sending {req_msg.__class__.__name__} to {dest} costs {timer.cost()}s, "
                     f"size={round(mb_size, 2)}MB, speed={round(mb_size / (timer.cost() + 1e-6), 2)}MB/s")
    if 'r' in mode:
        mb_size = rsp_msg.ByteSize() / 1024 / 1024  # 单位MB
        logger.debug(f"Getting {req_msg.__class__.__name__} from {dest} costs {timer.cost()}s, "
                     f"size={round(mb_size, 2)}MB, speed={round(mb_size / (timer.cost() + 1e-6), 2)}MB/s")
    return rsp_msg
