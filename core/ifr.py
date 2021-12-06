from dataclasses import dataclass
from typing import Dict, List

from torch import Tensor

from core.dif_executor import DifJob
from rpc.msg_pb2 import IFRMsg, WkJobMsg


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