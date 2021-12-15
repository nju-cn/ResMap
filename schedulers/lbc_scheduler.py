from typing import List, Dict, Any

from torch import Tensor

from core.ifr import WkJob
from core.predictor import Predictor
from master.scheduler import Scheduler, SizedNode


class LBCScheduler(Scheduler):
    def __init__(self, dag: List[SizedNode], predictors: List[Predictor], wk_costs: List[List[float]],
                 config: Dict[str, Any]):
        base_wk = 0  # 编号最小的作为计算能力的baseline
        self.__ly_comp = wk_costs[base_wk]  # 各层计算能力，以base_wk为基准
        self.__wk_cap = []  # worker_id->相对计算能力
        for wk, costs in enumerate(wk_costs):
            assert costs[0] == 0, f"InputModule of Worker{wk} cost should be 0!"
            # Worker计算能力：基准worker的总耗时 / 当前worker的总耗时
            self.__wk_cap.append(sum(wk_costs[base_wk]) / sum(costs))
        wk_lynum = self.split_chain(self.__ly_comp[1:], self.__wk_cap)  # 第0层不需要执行
        self.__lb_wk_elys = self.wk_lynum2layers_chain(1, wk_lynum)
        self.__job_type = config['job']

    def gen_wk_jobs(self, ifr_id: int, pre_ipt: Tensor, cur_ipt: Tensor) -> List[WkJob]:
        jobs = [WkJob(w, self.__job_type(lys, ([lys[-1]] if lys else []), {})) for w, lys in enumerate(self.__lb_wk_elys)]
        jobs[0].job.id2data[0] = cur_ipt - pre_ipt
        return jobs
