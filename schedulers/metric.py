from abc import abstractmethod
from typing import List


class Metric:
    def __init__(self, ly_comp: List[float], wk_cap: List[float], wk_bwth: List[float],
                 pre_wk_ilys: List[List[int]], org_gp_lbsz: List[List[float]], dif_gp_lbsz: List[List[float]]):
        self._ly_comp = ly_comp
        self._wk_cap = wk_cap
        assert len(wk_bwth) == len(pre_wk_ilys)
        self._wk_bwth = wk_bwth
        self._pre_wk_ilys = pre_wk_ilys
        assert len(org_gp_lbsz) == len(dif_gp_lbsz)
        self._org_gp_lbsz = org_gp_lbsz
        self._dif_gp_lbsz = dif_gp_lbsz

    @abstractmethod
    def __call__(self, gp_wk_elys: List[List[List[int]]]) -> float:
        """输入当前IFR组各个IFR的执行方案，返回方案的代价，越小越好
        :param gp_wk_elys: gp_wk_elys[g][w]为第g个IFR在Worker w上要执行的任务
        :return 相应的代价
        """

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
        # fwk之后，传输代价都是0
        for w in range(fwk + 1, len(wk_bwth)):
            wk_tran[w] = 0.
        return wk_tran

    @classmethod
    def plan2cmpt_chain(cls, wk_elys: List[List[int]], wk_cap: List[float], ly_comp: List[float]) -> List[float]:
        """输入执行计划，得到各Worker的计算耗时wk_cmpt. wk_cmpt[w]：单帧中，Worker w完成自己的所有层的计算耗时"""
        return [sum(ly_comp[e] for e in wk_elys[w]) / wk_cap[w] for w in range(len(wk_cap))]


class LatencyMetric(Metric):
    """指标为IFRGroup完成耗时"""
    def __call__(self, gp_wk_elys: List[List[List[int]]]) -> float:
        gp_wk_tran = []
        gp_wk_cmpt = []
        pre_wk_ilys = self._pre_wk_ilys
        for gi in range(len(self._org_gp_lbsz)):
            wk_tran = self.plan2tran_chain(gp_wk_elys[gi], pre_wk_ilys, self._wk_bwth,
                                                  self._org_gp_lbsz[gi], self._dif_gp_lbsz[gi])
            gp_wk_tran.append(wk_tran)
            pre_wk_ilys = [([lys[0]] if len(lys) > 0 else []) for lys in gp_wk_elys[gi]]
            wk_cmpt = self.plan2cmpt_chain(gp_wk_elys[gi], self._wk_cap, self._ly_comp)
            gp_wk_cmpt.append(wk_cmpt)
        fs_dp = self.simulate_pipeline(gp_wk_tran, gp_wk_cmpt)
        return fs_dp[-1][-1]

    @classmethod
    def simulate_pipeline(cls, gp_wk_tran: List[List[float]], gp_wk_cmpt: List[List[float]]) \
            -> List[List[float]]:
        """输入各帧各Worker的传输耗时gp_wk_tran，计算耗时gp_wk_cmpt，返回各帧在各Worker上的完成耗时。Worker可以有多个
        计算量为0的worker也会出现在fs_dp中
        """
        assert len(gp_wk_tran) > 0
        assert len(gp_wk_tran) == len(gp_wk_cmpt)
        assert all(len(wk_tran) == len(wk_cmpt) for wk_tran, wk_cmpt in zip(gp_wk_tran, gp_wk_cmpt))
        nframe = len(gp_wk_tran)
        nworker = len(gp_wk_tran[0])
        # dp[f][i]：第f帧第i//2个Worker的 传输完成耗时(i为偶数) / 计算完成耗时(i为奇数)
        # dp[f][2*w]：第f帧第w个Worker的传输完成耗时
        # dp[f][2*w+1]：第f帧第w个Worker的计算完成耗时
        dp = [[0. for _ in range(nworker * 2)] for _ in range(nframe)]
        # 第0帧的传输代价为gp_wk_tran[0], 计算代价为gp_wk_cmpt[0]
        dp[0][0] = gp_wk_tran[0][0]  # 传输完成
        dp[0][1] = gp_wk_tran[0][0] + gp_wk_cmpt[0][0]  # 计算完成
        for w in range(1, nworker):
            dp[0][2 * w] = dp[0][2 * w - 1] + gp_wk_tran[0][w]
            dp[0][2 * w + 1] = dp[0][2 * w] + gp_wk_cmpt[0][w]
        # 第1帧及其后面的传输代价为gp_wk_tran[f]，计算代价为gp_wk_cmpt[f]
        for f in range(1, nframe):
            dp[f][0] = dp[f - 1][0] + gp_wk_tran[f][0]  # Master发完了上一帧，开始发下一帧
            dp[f][1] = max(dp[f - 1][1], dp[f][0]) + gp_wk_cmpt[f][0]  # 上一帧的计算完成，且当前帧传输完成，才开始当前帧的计算
            for w in range(1, nworker):
                dp[f][2 * w] = max(dp[f - 1][2 * w], dp[f][2 * w - 1]) + gp_wk_tran[f][w]
                dp[f][2 * w + 1] = max(dp[f - 1][2 * w + 1], dp[f][2 * w]) + gp_wk_cmpt[f][w]
        return dp