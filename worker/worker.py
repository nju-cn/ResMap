import time
from queue import Queue
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from threading import Thread, Condition

import torch
import tqdm
from torch import Tensor

from core.dif_executor import DifJob, DifExecutor
from core.executor import Node
from core.integral_executor import IntegralExecutor, ExNode, IntegralJob
from core.util import cached_func, dnn_abbr
from rpc.msg_pb2 import IFRMsg, WkJobMsg, ResultMsg
from core.raw_dnn import RawDNN
from rpc.stub_factory import StubFactory


@dataclass
class WkDifJob:
    worker_id: int
    dif_job: DifJob

    @classmethod
    def from_msg(cls, wj_msg: WkJobMsg) -> 'WkDifJob':
        return cls(wj_msg.worker_id, DifJob.from_msg(wj_msg.job_msg))

    def to_msg(self) -> WkJobMsg:
        return WkJobMsg(worker_id=self.worker_id, job_msg=self.dif_job.to_msg())


@dataclass
class IFR:
    """一个帧对应的Worker执行计划，在Worker之间按照顺序传递
    当一帧到来时，Master生成每个Worker相应Job的exec_ids和out_ids
    每次传递时，把已完成的Job所有字段置为空，设置下一个Job的id2opt
    """
    id: int  # 帧号
    wk_jobs: List[WkDifJob]  # 按照执行顺序排列，至少有一个，不会为空

    @classmethod
    def from_msg(cls, ifr_msg: IFRMsg) -> 'IFR':
        return cls(ifr_msg.id, [WkDifJob.from_msg(wj_msg) for wj_msg in ifr_msg.wk_jobs])

    def to_msg(self) -> IFRMsg:
        return IFRMsg(id=self.id, wk_jobs=[job.to_msg() for job in self.wk_jobs])

    def is_final(self) -> bool:
        return len(self.wk_jobs) == 1

    def switch_next(self, id2dif: Dict[int, Tensor]) -> None:
        assert not self.is_final()
        self.wk_jobs.pop(0)
        self.wk_jobs[0].dif_job.id2dif = id2dif


class Worker(Thread):
    """以pipeline的方式执行Job"""
    def __init__(self, id_: int, stb_fct: StubFactory, config: Dict[str, Any]) -> None:
        super().__init__()
        self.__id = id_
        self.__config = config
        self.__cv = Condition()
        raw_dnn = RawDNN(config['dnn_loader']())
        self.__executor = DifExecutor(raw_dnn)
        self.__ex_queue: Queue[IFRMsg] = Queue()  # 执行的任务队列
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
            ifr = IFR.from_msg(self.__ex_queue.get())
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
                self.__stb_fct.worker(ifr.wk_jobs[0].worker_id).new_ifr(ifr.to_msg())
            else:
                print(f"IFR{ifr.id} finished")
                if self.__config['check']:
                    self.__stb_fct.master()\
                        .check_result(ResultMsg(ifr_id=ifr.id,
                                                arr3d=DifJob.tensor4d_arr3dmsg(
                                                    next(iter(self.__executor.last_out().values())))))

    def new_ifr(self, ifr_msg: IFRMsg) -> None:
        self.__ex_queue.put(ifr_msg)

    def profile_cost(self) -> List[float]:
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
