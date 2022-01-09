import copy
import logging
import operator
from functools import reduce
from typing import List, Type, Dict, Any

from torch import Tensor

from core.dif_executor import DifJob
from core.executor import Job
from core.ifr import IFR, WkJob
from core.predictor import Predictor
from master.scheduler import SizedNode, Scheduler


class MyScheduler(Scheduler):
    """以IFR Group为粒度进行调度"""
    def __init__(self, s_dag: List[SizedNode], predictors: List[Predictor],
                 wk_cap: List[float], wk_bwth: List[float], ly_comp: List[float],
                 job_type: Type[Job], ifr_num: int, config: Dict[str, Any]):
        assert len(wk_cap) == len(wk_bwth) == 2
        self.__sdag = s_dag
        self.__predictors = predictors
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
        # TODO: 无论原始数据是否稀疏，传输时都会尝试使用稀疏编码，这可能会导致估计的原始数据lbsz和实际的不符。
        #  那么传输的时候是否要指定不使用稀疏编码？（这里可能涉及到IFR序列化等地方的修改）
        #  需要观察一下原始数据的非零率。
        #  * 若很少稀疏则直接指定用原始数据，不稀疏编码；
        #  * 否则可能会影响motivation。可以看一下能否用predictor直接预测，或者其他办法
        org_gp_lbsz = [[reduce(operator.mul, snd.out_size)*4 for snd in self.__sdag] for ipt in ipt_group]
        dif_gp_lbsz = [Scheduler.dif2lbsz(dif, self.__sdag, self.__predictors) for dif in dif_group]
        opt_wk_elys, opt_cost = [], float('inf')
        # 遍历所有边，找到最优解
        for ly in range(1, len(self.__sdag)+1):
            # 设len(self.__sdag)=N, worker0执行dag[1:ly], worker1执行dag[ly:N]
            # ly=1时, w0执行[], w1执行dag[1:]; ly=N时, w0执行dag[1:N], w1执行[]
            wk_elys = [list(range(1, ly)), list(range(ly, len(self.__sdag)))]
            gp_wk_tran = []
            pre_wk_ilys = self.__pre_wk_ilys
            for gi in range(len(ipt_group)):
                wk_tran = self.plan2tran_chain(wk_elys, pre_wk_ilys, self.__wk_bwth, org_gp_lbsz[gi], dif_gp_lbsz[gi])
                gp_wk_tran.append(wk_tran)
                pre_wk_ilys = [([lys[0]] if len(lys) > 0 else []) for lys in wk_elys]
            wk_cmpt = self.plan2cmpt_chain(wk_elys, self.__wk_cap, self.__ly_comp)
            fs_dp = self.simulate_pipeline(gp_wk_tran, wk_cmpt)
            self.__logger.info(f"[1:{ly}]+[{ly}:{len(self.__sdag)}] => cost={fs_dp[-1][-1]}")
            if fs_dp[-1][-1] < opt_cost:
                opt_wk_elys, opt_cost = wk_elys, fs_dp[-1][-1]
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
    def plan2tran_chain(cls, wk_elys: List[List[int]], pre_wk_ilys: List[List[int]], wk_bwth: List[float],
                        org_lbsz: List[float], dif_lbsz: List[float]) -> List[float]:
        """输入当前帧的执行计划和执行前的缓存分布，得到各Worker的传输耗时wk_tran。只考虑链状CNN
        :param wk_elys 为本次的执行计划
        :param pre_wk_ilys 为上一次执行中，各Worker的输入层
        :param org_lbsz 为各层的原始数据大小
        :param dif_lbsz 为各层的差值数据大小
        :return 传输代价wk_tran. wk_tran[w]: Worker w接收前一个Worker输出数据的传输耗时
        """
        fwk = len(wk_elys) - 1  # 最后一个有执行任务的Worker
        while fwk > 0 and len(wk_elys[fwk]) == 0:
            fwk -= 1
        # wk_tran[w]：Worker w接收前一个Worker输出数据的传输耗时
        wk_tran = [0. for _ in range(len(wk_bwth))]  # 长度等于总worker数量
        wk_tran[0] = dif_lbsz[0] / wk_bwth[0]  # 第一个Worker无论是否执行第1层，都接收的是差值数据
        for w in range(1, fwk + 1):
            if len(wk_elys[w - 1]) == 0:  # Worker w-1不执行任何计算任务
                wk_tran[w] = wk_tran[w - 1] * wk_bwth[w - 1] / wk_bwth[w]  # Worker w的传输数据量和前一个Worker(w-1)一样
            else:
                if wk_elys[w - 1][-1] in pre_wk_ilys[w]:  # 输入数据已有缓存
                    bsz = dif_lbsz[wk_elys[w - 1][-1]]
                else:  # 输入数据没有缓存
                    bsz = org_lbsz[wk_elys[w - 1][-1]]
                wk_tran[w] = bsz / wk_bwth[w]  # Worker w-1的输出数据量/带宽
        for w in range(fwk + 1, len(wk_bwth)):
            wk_tran[w] = 0.
        return wk_tran

    @classmethod
    def plan2cmpt_chain(cls, wk_elys: List[List[int]], wk_cap: List[float], ly_comp: List[float]) -> List[float]:
        """输入执行计划，得到各Worker的计算耗时wk_cmpt. wk_cmpt[w]：单帧中，Worker w完成自己的所有层的计算耗时"""
        return [sum(ly_comp[e] for e in wk_elys[w]) / wk_cap[w] for w in range(len(wk_cap))]

    @classmethod
    def simulate_pipeline(cls, gp_wk_tran: List[List[float]], wk_cmpt: List[float]) -> List[List[float]]:
        """输入各帧各Worker的传输耗时gp_wk_tran，计算耗时wk_cmpt，返回各帧在各Worker上的完成耗时。Worker可以有多个
        注意：返回的fs_dp中的worker数<=len(wk_cmpt)=len(wk_tran)，因为不包括后面的计算量为0的worker
        """
        assert len(gp_wk_tran) > 0
        assert all(len(wk_tran) == len(wk_cmpt) for wk_tran in gp_wk_tran)
        nframe = len(gp_wk_tran)
        act_nwk = len(wk_cmpt)  # 从Worker0开始，到最后一个计算量不为0的Worker，实际总共有几个Worker参与计算
        while act_nwk > 0 and wk_cmpt[act_nwk - 1] == 0:
            act_nwk -= 1
        assert act_nwk > 0, "There is no worker to compute!"
        # dp[f][i]：第f帧第i//2个Worker的 传输完成耗时(i为偶数) / 计算完成耗时(i为奇数)
        # dp[f][2*w]：第f帧第w个Worker的传输完成耗时
        # dp[f][2*w+1]：第f帧第w个Worker的计算完成耗时
        dp = [[0. for _ in range(act_nwk * 2)] for _ in range(nframe)]
        # 第0帧的传输代价为gp_wk_tran[0]
        dp[0][0] = gp_wk_tran[0][0]  # 传输完成
        dp[0][1] = gp_wk_tran[0][0] + wk_cmpt[0]  # 计算完成
        for w in range(1, act_nwk):
            dp[0][2 * w] = dp[0][2 * w - 1] + gp_wk_tran[0][w]
            dp[0][2 * w + 1] = dp[0][2 * w] + wk_cmpt[w]
        # 第1帧及其后面的传输代价为gp_wk_tran[f]
        for f in range(1, nframe):
            dp[f][0] = dp[f - 1][0] + gp_wk_tran[f][0]  # Master发完了上一帧，开始发下一帧
            dp[f][1] = max(dp[f - 1][1], dp[f][0]) + wk_cmpt[0]  # 上一帧的计算完成，且当前帧传输完成，才开始当前帧的计算
            for w in range(1, act_nwk):
                dp[f][2 * w] = max(dp[f - 1][2 * w], dp[f][2 * w - 1]) + gp_wk_tran[f][w]
                dp[f][2 * w + 1] = max(dp[f - 1][2 * w + 1], dp[f][2 * w]) + wk_cmpt[w]
        return dp
