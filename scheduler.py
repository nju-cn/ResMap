from typing import List

from dif_executor import DifJob
from node import RNode
from worker import IFR, WkDifJob


class Scheduler:
    def __init__(self, r_dag: List[RNode]):
        self.__r_dag = r_dag

    def gen_wk_jobs(self) -> List[WkDifJob]:
        return [WkDifJob(0, DifJob(list(range(1, 14)), [13], {}))]
