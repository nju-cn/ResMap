from typing import List, Optional, Tuple

import torch
from torch import Tensor

from core.dif_executor import DifJob
from core.executor import Node
from core.integral_executor import ExNode, IntegralExecutor, IntegralJob
from core.predictor import Predictor
from core.raw_dnn import RawDNN
from worker.worker import WkDifJob


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
    def raw2dag_sized(cls, raw_dnn: RawDNN, frame_size: Tuple[int, int]) -> List['SizedNode']:
        """使用RawDNN和指定的帧大小，初始化保存有输出数据大小的DAG图"""
        itg_extor = IntegralExecutor(raw_dnn, _SizingExNode)
        ipt = torch.rand(1, 3, *frame_size)
        job = IntegralJob(list(range(1, len(raw_dnn.layers))), [raw_dnn.layers[-1].id_], {0: ipt})
        itg_extor.exec(job)
        return [cls(se_node) for se_node in itg_extor.dag()]


class Scheduler:
    def __init__(self, dag: List[SizedNode], predictors: List[Predictor]):
        self.__dag = dag
        self.__predictors = predictors

    def gen_wk_jobs(self, dif_ipt: Tensor) -> List[WkDifJob]:
        cnz = [float(chan.count_nonzero()/chan.nelement()) for chan in dif_ipt[0]]
        lcnz = self.predict_dag(cnz, self.__dag, self.__predictors)
        # print(lcnz)
        return [WkDifJob(0, DifJob(list(range(1, 5)), [4], {0: dif_ipt})),
                WkDifJob(1, DifJob(list(range(5, 10)), [9], {})),
                WkDifJob(2, DifJob(list(range(10, 14)), [13], {}))]

    @classmethod
    def predict_dag(cls, ipt_nz: List[float], dag: List[Node], predictors: List[Predictor]) -> List[List[float]]:
        """根据输入数据与上一帧的非零占比，预测DAG各个节点输出数据与上一帧的非零占比"""
        assert len(dag) == len(predictors)
        results = [[] for _ in range(len(dag))]
        results[0] = predictors[0].predict([ipt_nz])
        for d in dag[0].descendants:
            cls._predict_dag(d, results, dag, predictors)
        return results

    @classmethod
    def _predict_dag(cls, node_id: int, res_lcnz: List[List[float]],
                     dag: List[Node], predictors: List[Predictor]) -> None:
        """模仿core.raw_dnn.RawDNN.__execute_dag
        res_lcnz的每个元素必须初始化为空列表
        """
        if len(res_lcnz[node_id]) > 0:
            return
        acnz = []
        for aid in dag[node_id].ancients:
            if len(res_lcnz[aid]) > 0:
                acnz.append(res_lcnz[aid])
            else:
                return
        res_lcnz[node_id] = predictors[node_id].predict(acnz)
        for d in dag[node_id].descendants:
            cls._predict_dag(d, res_lcnz, dag, predictors)
