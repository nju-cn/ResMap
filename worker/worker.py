import time
from queue import Queue
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from threading import Thread, Condition

import torch
import tqdm
from torch import Tensor

from core.dif_executor import DifExecutor
from core.executor import Node
from core.ifr import IFR
from core.integral_executor import IntegralExecutor, ExNode, IntegralJob
from core.util import cached_func, dnn_abbr
from core.raw_dnn import RawDNN
from rpc.stub_factory import StubFactory


class Worker(Thread):
    """以pipeline的方式执行Job"""
    def __init__(self, id_: int, stb_fct: StubFactory, config: Dict[str, Any]) -> None:
        super().__init__()
        self.__id = id_
        self.__config = config
        self.__cv = Condition()
        raw_dnn = RawDNN(config['dnn_loader']())
        self.__executor = DifExecutor(raw_dnn)
        self.__ex_queue: Queue[IFR] = Queue()  # 执行的任务队列
        self.__stb_fct = stb_fct
        print(f"Worker{self.__id} profiling...")
        self.__costs = []
        costs = cached_func(f"w{id_}.{dnn_abbr(config['dnn_loader'])}.cst", self.profile_dnn_cost,
                            raw_dnn, config['frame_size'], config['worker']['prof_niter'])
        with self.__cv:
            self.__costs = costs
            self.__cv.notifyAll()

    def id(self):
        return self.__id

    def run(self) -> None:
        last_ifr_id = -1
        while True:
            ifr = self.__ex_queue.get()
            print(f"get IFR{ifr.id}")
            assert ifr.id == last_ifr_id + 1, "IFR sequence is inconsistent, DifJob cannot be executed!"
            assert len(ifr.wk_jobs) > 0, "IFR has finished, cannot be executed!"
            assert ifr.wk_jobs[0].worker_id == self.__id, \
                f"IFR(wk={ifr.wk_jobs[0].worker_id}) should not appear in Worker{self.__id}!"
            id2dif = self.__executor.exec(ifr.wk_jobs[0].dif_job)
            print(f"execute IFR{ifr.id}: {ifr.wk_jobs[0].dif_job.exec_ids}")
            last_ifr_id = ifr.id
            if not ifr.is_final():
                ifr.switch_next(id2dif)
                self.__stb_fct.worker(ifr.wk_jobs[0].worker_id).new_ifr(ifr)
            else:
                print(f"IFR{ifr.id} finished")
                if self.__config['check']:
                    result = next(iter(self.__executor.last_out().values()))
                else:
                    result = None
                self.__stb_fct.master().report_finish(ifr.id, result)

    def new_ifr(self, ifr: IFR) -> None:
        self.__ex_queue.put(ifr)

    def layer_cost(self) -> List[float]:
        """执行配置文件指定的CNN，返回每层耗时"""
        with self.__cv:
            while len(self.__costs) == 0:
                self.__cv.wait()
            print("got layer costs")
            return self.__costs

    class _TimingExNode(ExNode):
        def __init__(self, node: Node):
            super().__init__(node)
            self.cost = 0

        def execute(self, *inputs: Tensor) -> None:
            begin = time.time()
            super().execute(*inputs)
            self.cost = time.time() - begin

    @classmethod
    def profile_dnn_cost(cls, raw_dnn: RawDNN, frame_size: Tuple[int, int], niter: int) -> List[float]:
        itg_extor = IntegralExecutor(raw_dnn, cls._TimingExNode)
        ipt = torch.rand(1, 3, *frame_size)
        job = IntegralJob(list(range(1, len(raw_dnn.layers))), [raw_dnn.layers[-1].id_], {0: ipt})
        layer_cost = [0 for _ in range(len(raw_dnn.layers))]
        for _ in tqdm.tqdm(range(niter)):
            itg_extor.exec(job)
            for pnode in itg_extor.dag():
                layer_cost[pnode.id] += pnode.cost
        for l in range(len(raw_dnn.layers)):
            layer_cost[l] /= niter
        return layer_cost
