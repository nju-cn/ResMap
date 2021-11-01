from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

from torch import Tensor


@dataclass
class Job:
    """供Executor使用的通用接口"""
    exec_ids: List[int]  # 要执行的这组CNN层的id，按照执行顺序排列
    out_ids: List[int]  # 这组CNN层中输出层的id


class Executor(ABC):
    """抽象类，用于定义执行Job的统一接口"""
    @abstractmethod
    def exec(self, job: Job) -> Dict[int, Tensor]:
        """送入单个任务，执行并得到输出
        :return {node_id: 数据}
        """
        pass

    @abstractmethod
    def check_exec(self, input_: Tensor) -> List[Tensor]:
        """执行给定输入，返回各层的输出"""
        pass
