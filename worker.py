from queue import Queue
from dataclasses import dataclass
from typing import Dict, Callable, Any, List, Tuple
from threading import Thread

from torch import Tensor

from dnn_config import DNNConfig
from dif_executor import DifJob, DifExecutor
from msg_pb2 import IFRMsg, WkJobMsg, ResultMsg


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
    wk_jobs: List[WkDifJob]  # 按照执行顺序排列

    @classmethod
    def from_msg(cls, ifr_msg: IFRMsg) -> 'IFR':
        return cls(ifr_msg.id, [WkDifJob.from_msg(wj_msg) for wj_msg in ifr_msg.wk_jobs])

    def to_msg(self) -> IFRMsg:
        return IFRMsg(id=self.id, wk_jobs=[job.to_msg() for job in self.wk_jobs])

    def switch_next(self, id2dif: Dict[int, Tensor]) -> None:
        assert len(self.wk_jobs) > 1
        self.wk_jobs.pop(0)
        self.wk_jobs[0].dif_job.id2dif = id2dif


class Worker(Thread):
    """以pipeline的方式执行Job"""
    def __init__(self, id_: int, dnn_loader: Callable[[], DNNConfig], check: bool,
                 send_ifr_async: Callable[[IFRMsg], None],
                 check_result: Callable[[ResultMsg], None]) -> None:
        super().__init__()
        self.__id = id_
        self.__check = check
        self.__executor = DifExecutor(dnn_loader)
        self.__ex_queue: Queue[IFRMsg] = Queue()  # 执行的任务队列
        self.__send_ifr_async = send_ifr_async
        self.__check_result = check_result

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
            if len(ifr.wk_jobs) > 1:
                ifr.switch_next(id2dif)
                self.__send_ifr_async(ifr.to_msg())
            else:
                print(f"IFR{ifr.id} finished")
                if self.__check:
                    self.__check_result(ResultMsg(ifr_id=ifr.id,
                                                  arr3d=DifJob.tensor4d_arr3dmsg(
                                                      next(iter(self.__executor.last_out().values())))))

    def new_ifr(self, ifr_msg: IFRMsg) -> None:
        self.__ex_queue.put(ifr_msg)
