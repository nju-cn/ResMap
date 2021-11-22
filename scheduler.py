from typing import List, Optional, Tuple

import torch
from torch import Tensor

from dif_executor import DifJob
from executor import Node
from integral_executor import ExNode, IntegralExecutor, IntegralJob
from predictor import Predictor
from raw_dnn import RawDNN
from worker import WkDifJob


class _SizingExNode(ExNode):
    """在Executor中运行，以获取输出数据大小的信息"""
    def __init__(self, node: Node):
        super().__init__(node)
        self.out_size = None

    def set_finish(self, output: Optional[Tensor]) -> None:
        super().set_finish(output)
        self.out_size = tuple(self.get_output().shape)[1:]

    def execute(self, *inputs: Tensor) -> None:
        super().execute(*inputs)
        self.out_size = tuple(self.get_output().shape)[1:]


class SizedNode(Node):
    """根据输出数据大小和稀疏率预测模型进行初始化"""
    def __init__(self, se_node: _SizingExNode):
        super().__init__(se_node.id, se_node.ancients, se_node.descendants, se_node.calc)
        self.out_size: Tuple[int, int, int] = se_node.out_size  # (通道数, 行数, 列数)

    @classmethod
    def raw2dag(cls, raw_dnn: RawDNN, frame_size: Tuple[int, int]) -> List['SizedNode']:
        """使用RawDNN和指定的帧大小，初始化保存有输出数据大小的DAG图"""
        itg_extor = IntegralExecutor(raw_dnn, _SizingExNode)
        ipt = torch.rand(1, 3, *frame_size)
        job = IntegralJob(list(range(1, len(raw_dnn.layers))), [raw_dnn.layers[-1].id_], {0: ipt})
        itg_extor.exec(job)
        return [cls(se_node) for se_node in itg_extor.dag()]


class Scheduler:
    def __init__(self, dag: List[SizedNode], predictors: List[Predictor]):
        self.__dag = dag

    def gen_wk_jobs(self) -> List[WkDifJob]:
        return [WkDifJob(0, DifJob(list(range(1, 5)), [4], {})),
                WkDifJob(1, DifJob(list(range(5, 10)), [9], {})),
                WkDifJob(2, DifJob(list(range(10, 14)), [13], {}))]
