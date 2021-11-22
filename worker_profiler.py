import time
from typing import Tuple, List

import torch
from torch import Tensor
import tqdm


from dnn_models.chain import prepare_alexnet
from integral_executor import ExNode, IntegralExecutor, IntegralJob
from executor import Node
from raw_dnn import RawDNN


class _TimingExNode(ExNode):
    def __init__(self, node: Node):
        super().__init__(node)
        self.cost = 0

    def execute(self, *inputs: Tensor) -> None:
        begin = time.time()
        super(_TimingExNode, self).execute(*inputs)
        self.cost = time.time() - begin


class WorkerProfiler:
    def __init__(self, raw_dnn: RawDNN, frame_size: Tuple[int, int], niter: int):
        self.__raw_dnn = raw_dnn
        self.__frame_size = frame_size
        self.__niter = niter

    def profile(self) -> List[float]:
        """对当前Worker执行各CNN的计算能力进行测试"""
        itg_extor = IntegralExecutor(self.__raw_dnn, _TimingExNode)
        ipt = torch.rand(1, 3, *self.__frame_size)
        job = IntegralJob(list(range(1, len(self.__raw_dnn.layers))), [self.__raw_dnn.layers[-1].id_], {0: ipt})
        layer_cost = [0 for _ in range(len(self.__raw_dnn.layers))]
        for _ in tqdm.tqdm(range(self.__niter)):
            itg_extor.exec(job)
            for pnode in itg_extor.dag():
                layer_cost[pnode.id] += pnode.cost
        for l in range(len(self.__raw_dnn.layers)):
            layer_cost[l] /= self.__niter
        return layer_cost


if __name__ == '__main__':
    ly_cost = WorkerProfiler(RawDNN(prepare_alexnet()), (270, 480), 10).profile()
    print(ly_cost)
