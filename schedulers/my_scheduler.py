import copy
import logging
from typing import List, Type, Dict, Any, Callable, Tuple

from torch import Tensor

from core.dif_executor import DifJob
from core.executor import Job
from core.ifr import IFR, WkJob
from core.predictor import NZPred
from master.scheduler import SizedNode, Scheduler
from schedulers.metric import LatencyMetric, Metric


class MyScheduler(Scheduler):
    """以IFR Group为粒度进行调度"""
    # TODO: 改进！
    def __init__(self, s_dag: List[SizedNode], nzpred: NZPred,
                 wk_cap: List[float], wk_bwth: List[float], ly_comp: List[float],
                 job_type: Type[Job], ifr_num: int, config: Dict[str, Any]):
        assert len(wk_cap) == len(wk_bwth) == 2
        self.__sdag = s_dag
        artery = self.get_artery(self.__sdag)
        assert len(artery) == len(s_dag), "This scheduler is only used for chain!"
        self.__o_lbsz = [sz * 4 for sz in self.lcnz2lsz(nzpred.o_lcnz, s_dag)]
        self.__predictors = nzpred.predictors
        self.__wk_cap = wk_cap
        self.__wk_bwth = wk_bwth
        self.__ly_comp = ly_comp
        assert job_type == DifJob, "This scheduler is only used for DifJob!"
        self.__job_type: Type[Job] = job_type
        self.__gp_size: int = config['gp_size']

        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__pre_wk_ilys = [[] for _ in range(len(wk_cap))]  # 各Worker上次运行时的输入层

    def group_size(self) -> int:
        """建议的group大小，但实际上可能比这个小"""
        return self.__gp_size

    def gen_ifr_group(self, ifr_cnt: int, pre_ipt: Tensor, ipt_group: List[Tensor]) -> List[IFR]:
        """一个group的所有ifr都用同一个执行方案。ifr_cnt为当前group中第一个IFR的id
        注意：这里只考虑链状CNN，只考虑云边协同(只有两个Worker)
        """
        assert len(ipt_group) > 0
        dif_group = [ipt_group[0] - pre_ipt] + [ipt_group[i] - ipt_group[i-1] for i in range(1, len(ipt_group))]
        # 观察发现，原始数据也存在一定的稀疏性，但是分布非常集中。对于一个层而言，几乎所有帧的非零占比都在平均值附近
        # 所以这里直接使用平均值作为原始数据的非零率，进而计算原始数据大小
        org_gp_lbsz = [self.__o_lbsz for ipt in ipt_group]
        self.__logger.info(f"start predicting...")
        dif_gp_lbsz = [Scheduler.dif2lbsz(dif, self.__sdag, self.__predictors) for dif in dif_group]
        metric = LatencyMetric(self.__ly_comp, self.__wk_cap, self.__wk_bwth,
                               self.__pre_wk_ilys, org_gp_lbsz, dif_gp_lbsz)
        # TODO: 整理代码
        opt_wk_elys, opt_cost = self.recur_find_chain([], metric)
        # opt_wk_elys, opt_cost = [], float('inf')
        # for ly in range(1, len(self.__sdag)+1):
        #     # 设len(self.__sdag)=N, worker0执行dag[1:ly], worker1执行dag[ly:N]
        #     # ly=1时, w0执行[], w1执行dag[1:]; ly=N时, w0执行dag[1:N], w1执行[]
        #     wk_elys = [list(range(1, ly)), list(range(ly, len(self.__sdag)))]
        #     cost = metric([wk_elys for _ in ipt_group])
        #     self.__logger.info(f"[1,{ly-1}]+[{ly},{len(self.__sdag)-1}] => cost={cost}")
        #     if cost < opt_cost:
        #         opt_wk_elys, opt_cost = wk_elys, cost
        #     # TODO: Scheduler增加可视化接口，以便调试
        self.__logger.info(f"opt: {opt_wk_elys} => cost={opt_cost}")
        wk_jobs = [WkJob(w, self.__job_type(lys, ([lys[-1]] if lys else []), {})) for w, lys in enumerate(opt_wk_elys)]
        # Worker0接收到的输入数据必定为dif
        ifr_group = []
        for gi, dif in enumerate(dif_group):
            jobs = copy.deepcopy(wk_jobs)
            jobs[0].job.id2data = {0: dif}
            ifr_group.append(IFR(ifr_cnt + gi, jobs))
        return ifr_group

    @classmethod
    def recur_find_chain(cls, wk_elys: List[List[int]], metric: Metric) -> Tuple[List[List[int]], float]:
        """递归为各个Worker分配任务，根据metric寻找最优分配方案。只针对链状CNN
        :param wk_elys: 各Worker分配的层，初始为空，每次递归加一个Worker，Worker内各层按照id顺序排列
        :param wk_num: Worker数量
        :param ly_num: CNN层数
        :param metric: wk_elys的相应代价，越小越好
        :return 最优的wk_elys, 相应的代价
        """
        wk_num, ly_num = metric.wk_num(), metric.ly_num()
        worker_id = len(wk_elys)  # 当前要分配的Worker的ID
        last_ly = max((ly for lys in wk_elys for ly in lys), default=0)  # 当前Worker前已经完成的层
        if worker_id == wk_num-1:  # 当前要分配的是最后一个Worker
            # 最后一个Worker必须完成区间(last_ly, ly_num)的所有层
            wk_elys.append(list(range(last_ly+1, ly_num)))
            cost = metric([wk_elys]*metric.gp_size())
            res = copy.deepcopy(wk_elys)
            wk_elys.pop()
            return res, cost
        opt_wk_elys, opt_cost = [], float('inf')
        # my_last: 当前Worker完成之后，已经完成的层
        for my_last in range(last_ly, ly_num):  # last_ly表示当前Worker什么都没做，ly_num-1表示完成了剩余所有层
            wk_elys.append(list(range(last_ly+1, my_last+1)))
            cad_wk_elys, cad_cost = cls.recur_find_chain(wk_elys, metric)
            wk_elys.pop()
            if cad_cost < opt_cost:
                opt_wk_elys, opt_cost = cad_wk_elys, cad_cost
        assert opt_cost < float('inf')
        return opt_wk_elys, opt_cost
