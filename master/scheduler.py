import logging
import sys
from typing import List, Optional, Tuple, Dict, Any, Type

import torch
from torch import Tensor

from core.executor import Node, Job
from core.ifr import WkJob
from core.itg_executor import ExNode, ItgExecutor, ItgJob
from core.predictor import Predictor
from core.raw_dnn import RawDNN


class _SizingExNode(ExNode):
    """在Executor中运行，以获取输出数据大小的信息"""
    def __init__(self, node: Node):
        super().__init__(node)
        self.out_size = None

    def set_finish(self, output: Optional[Tensor]) -> None:
        super().set_finish(output)
        self.out_size = tuple(self.get_output().shape)[1:]

    def execute(self, *inputs: Tensor) -> None:
        super().execute(*inputs)
        self.out_size = tuple(self.get_output().shape)[1:]


class SizedNode(Node):
    """根据输出数据大小和稀疏率预测模型进行初始化"""
    def __init__(self, se_node: _SizingExNode):
        super().__init__(se_node.id, se_node.ancients, se_node.descendants, se_node.calc)
        self.out_size: Tuple[int, int, int] = se_node.out_size  # (通道数, 行数, 列数)
        _, R, C = se_node.out_size
        self.nz_thres = (1 - 1/C - 1/(R*C))/2  # 一个通道的非零占比<nz_thres时，稀疏压缩的数据传输量更少

    @classmethod
    def raw2dag_sized(cls, raw_dnn: RawDNN, frame_size: Tuple[int, int]) -> List['SizedNode']:
        """使用RawDNN和指定的帧大小，初始化保存有输出数据大小的DAG图"""
        itg_extor = ItgExecutor(raw_dnn, _SizingExNode)
        ipt = torch.rand(1, 3, *frame_size)
        job = ItgJob(list(range(1, len(raw_dnn.layers))), [raw_dnn.layers[-1].id_], {0: ipt})
        itg_extor.exec(job)
        return [cls(se_node) for se_node in itg_extor.dag()]


class Scheduler:
    def __init__(self, dag: List[SizedNode], predictors: List[Predictor],
                 wk_costs: List[List[float]], config: Dict[str, Any]):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__config = config
        self.__dag = dag
        self.__predictors = predictors
        base_wk = 0  # 编号最小的作为计算能力的baseline
        self.__logger.debug(f"baseline is worker{base_wk}")
        self.__ly_comp = wk_costs[base_wk]  # 各层计算能力，以base_wk为基准
        self.__wk_cap = []  # worker_id->相对计算能力
        self.__job_type: Type[Job] = config['job']
        for wk, costs in enumerate(wk_costs):
            assert costs[0] == 0, f"InputModule of Worker{wk} cost should be 0!"
            # Worker计算能力：基准worker的总耗时 / 当前worker的总耗时
            self.__wk_cap.append(sum(wk_costs[base_wk]) / sum(costs))
        self.__logger.debug(f"wk_cap={self.__wk_cap}")
        self.__logger.debug(f"ly_comp={self.__ly_comp}")
        wk_lynum = self.split_chain(self.__ly_comp[1:], self.__wk_cap)  # 第0层不需要执行
        self.__lb_wk_elys = self.wk_lynum2layers_chain(1, wk_lynum)
        self.__logger.debug(f"load balance: {self.__lb_wk_elys}")
        self.__wk_bwth = [bw*1024*1024 for bw in config['master']['scheduler']['bandwidth']]  # 单位MB转成B

    def gen_wk_jobs(self, dif_ipt: Tensor) -> List[WkJob]:
        cnz = [float(chan.count_nonzero()/chan.nelement()) for chan in dif_ipt[0]]
        lcnz = self.predict_dag(cnz, self.__dag, self.__predictors)
        lsz = self.lcnz2lsz(lcnz, self.__dag)
        lbsz = [sz*4 for sz in lsz]
        wk_layers = self.optimize_chain(self.__lb_wk_elys, self.__wk_cap, self.__wk_bwth,
                                        lbsz, self.__ly_comp, self.__config['master']['ifr_num'],
                                        vis=False, logger=self.__logger)
        jobs = [WkJob(w, self.__job_type(lys, ([lys[-1]] if lys else []), {})) for w, lys in enumerate(wk_layers)]
        jobs[0].job.id2data[0] = dif_ipt
        return jobs

    @classmethod
    def predict_dag(cls, ipt_nz: List[float], dag: List[Node], predictors: List[Predictor]) -> List[List[float]]:
        """根据输入数据与上一帧的非零占比，预测DAG各个节点输出数据与上一帧的非零占比"""
        assert len(dag) == len(predictors)
        results = [[] for _ in range(len(dag))]
        results[0] = predictors[0].predict([ipt_nz])
        for d in dag[0].descendants:
            cls._predict_dag(d, results, dag, predictors)
        return results

    @classmethod
    def lcnz2lsz(cls, lcnz: List[List[float]], s_dag: List[SizedNode]) -> List[float]:
        """对各层，根据通道的非零占比计算出输出数据总元素个数"""
        lsz = []
        for l in range(len(s_dag)):
            size = 0
            H, R, C = s_dag[l].out_size
            for c in range(H):
                p = lcnz[l][c]
                if p < s_dag[l].nz_thres:
                    size += 2 * R * C * p + R + 1
                else:
                    size += R * C
            lsz.append(size)
        return lsz

    @classmethod
    def split_chain(cls, ly_comp: List[float], wk_cap: List[float]) -> List[int]:
        """按照Worker的计算能力，对链状的CNN进行切割使得各Worker耗时相近，返回切割点（切割点属于前一个Worker）
        :param ly_comp: ly_comp[l]为第l层的计算量，即baseline的worker运行耗时
        :param wk_cap: 按照Worker执行顺序，各Worker的相对计算能力，其中一个worker的计算能力为1，作为baseline
        :return: 各Worker执行的层数
        """
        assert len(ly_comp) > 0, "The number of layers is 0!"
        assert len(wk_cap) > 0, "There is no worker!"
        ly_comp_acc = [0. for _ in range(len(ly_comp)+1)]  # ly_comp的累积值: ly_comp_acc[l] = sum(ly_comp[:l])
        for l in range(len(ly_comp)):
            ly_comp_acc[l+1] = ly_comp_acc[l] + ly_comp[l]
        total_comp, total_cap = sum(ly_comp), sum(wk_cap)
        wk_comp = [cap / total_cap * total_comp for cap in wk_cap]  # 各worker应该分得的总计算量
        wk_comp_acc = [0. for _ in range(len(wk_comp)+1)]  # wk_comp的累积值: wk_comp_acc[l] = sum(wk_comp[:l])
        for w in range(len(wk_comp)):
            wk_comp_acc[w+1] = wk_comp_acc[w] + wk_comp[w]
        # 按照Worker的执行顺序，每个Worker及其前驱执行的最后一个层,
        #   若w-1和w的层相同则表示w没有执行任何层，取值-1表示没有执行任何一个层
        wk_ly = []
        lycnt = 0  # 上一个worker执行的最后一个层+1
        for w, acc in enumerate(wk_comp_acc[1:-1]):  # acc[0]不对应任何worker；不考虑最后一个worker，因为它肯定是最后一个层
            ly = -100  # ly可能为-1，此时表示当前没有执行任何一个层
            while ly == -100:
                if lycnt == len(ly_comp_acc):  # 所有层都被前面的worker分配掉了
                    ly = len(ly_comp_acc)-1
                elif ly_comp_acc[lycnt] <= acc <= ly_comp_acc[lycnt+1]:  # lycnt处于边界上
                    if acc - ly_comp_acc[lycnt] <= ly_comp_acc[lycnt+1] - acc:
                        ly = lycnt-1
                    else:
                        ly = lycnt
                        lycnt += 1
                else:
                    lycnt += 1
            wk_ly.append(ly)
        wk_ly.append(len(ly_comp)-1)
        # wk_ly转成wk_lynum，表示各Worker执行的层数
        wk_lynum = [wk_ly[0]+1] + [wk_ly[w]-wk_ly[w-1] for w in range(1, len(wk_ly))]
        return wk_lynum

    @classmethod
    def wk_lynum2layers_chain(cls, begin_layer: int, wk_lynum: List[int]) -> List[List[int]]:
        """根据各Worker执行层数，从begin_layer开始，按照执行顺序为Worker分配具体执行的层。只考虑链状CNN
        :param begin_layer: Worker的任务从第几层开始，包括begin_layer
        :param wk_lynum: 各worker的层数
        :return: 各worker具体执行哪几层
        """
        ly_cnt = begin_layer
        wk_layers = []
        for lynum in wk_lynum:
            wk_layers.append(list(range(ly_cnt, ly_cnt+lynum)))
            ly_cnt += lynum
        return wk_layers

    @classmethod
    def plan2costs_chain(cls, wk_elys: List[List[int]], wk_cap: List[float], wk_bwth: List[float],
                         lbsz: List[float], ly_comp: List[float]) -> Tuple[List[float], List[float]]:
        """输入执行计划，得到各Worker的传输耗时wk_tran和计算耗时wk_cmpt。只考虑链状CNN
        :return wk_tran, wk_cmpt
        """
        # wk_tran[w]：单帧中，Worker w接收前一个Worker输出数据的传输耗时
        wk_tran: List[float] = [lbsz[0] / wk_bwth[0]]
        for w in range(1, len(wk_cap)):
            if len(wk_elys[w - 1]) == 0:  # Worker w-1不执行任何计算任务
                wk_tran.append(wk_tran[-1]*wk_bwth[w-1]/wk_bwth[w])  # Worker w的传输数据量和前一个Worker(w-1)一样
            else:
                wk_tran.append(lbsz[wk_elys[w - 1][-1]] / wk_bwth[w])  # Worker w-1的输出数据量/带宽
        # wk_cmpt[w]：单帧中，Worker w完成自己的所有层的计算耗时
        wk_cmpt = [sum(ly_comp[e] for e in wk_elys[w]) / wk_cap[w] for w in range(len(wk_cap))]
        return wk_tran, wk_cmpt

    @classmethod
    def simulate_pipeline(cls, wk_tran: List[float], wk_cmpt: List[float], nframe: int) -> List[List[float]]:
        """输入各Worker的传输耗时wk_tran，计算耗时wk_cmpt，帧数nframe，返回各帧在各Worker上的完成耗时
        注意：返回的fs_dp中的worker数<=len(wk_cmpt)=len(wk_tran)，因为不包括后面的计算量为0的worker
        """
        assert len(wk_tran) == len(wk_cmpt), f"len(wk_tran)<{len(wk_tran)}> != len(wk_cmpt)<{len(wk_cmpt)}>"
        act_nwk = len(wk_cmpt)  # 从Worker0开始，到最后一个计算量不为0的Worker，实际总共有几个Worker参与计算
        while act_nwk > 0 and wk_cmpt[act_nwk-1] == 0:
            act_nwk -= 1
        assert act_nwk > 0, "There is no worker to compute!"
        # dp[f][i]：第f帧第i//2个Worker的 传输完成耗时(i为偶数) / 计算完成耗时(i为奇数)
        # dp[f][2*w]：第f帧第w个Worker的传输完成耗时
        # dp[f][2*w+1]：第f帧第w个Worker的计算完成耗时
        dp = [[0. for _ in range(act_nwk*2)] for _ in range(nframe)]
        dp[0][0] = wk_tran[0]  # 传输完成
        dp[0][1] = wk_tran[0] + wk_cmpt[0]  # 计算完成
        for w in range(1, act_nwk):
            dp[0][2*w] = dp[0][2*w-1] + wk_tran[w]
            dp[0][2*w+1] = dp[0][2*w] + wk_cmpt[w]
        for f in range(1, nframe):
            dp[f][0] = dp[f-1][0] + wk_tran[0]  # Master发完了上一帧，开始发下一帧
            dp[f][1] = max(dp[f-1][1], dp[f][0]) + wk_cmpt[0]  # 上一帧的计算完成，且当前帧传输完成，才开始当前帧的计算
            for w in range(1, act_nwk):
                dp[f][2*w] = max(dp[f-1][2*w], dp[f][2*w-1]) + wk_tran[w]
                dp[f][2*w+1] = max(dp[f-1][2*w+1], dp[f][2*w]) + wk_cmpt[w]
        return dp

    @classmethod
    def estimate_latency_chain(cls, wk_elys: List[List[int]], wk_cap: List[float], wk_bwth: List[float],
                               lbsz: List[float], ly_comp: List[float], nframe: int, vis: bool = False) -> float:
        """输入计划任务和这个任务会执行的帧数，计算按照此计划任务执行的平均时延。只考虑链状CNN
        :param wk_elys 各worker执行哪些层，wk_elys[w][-1]就是Worker w的输出层
        :param wk_cap worker的计算能力
        :param wk_bwth wk_bwth[w]为w-1与w之间的带宽。w=0时为master和worker0之间的带宽
        :param lbsz 各层输出数据的大小，单位byte
        :param ly_comp 各layer的计算量
        :param nframe 总共会执行多少帧
        :param vis 是否对当前方案进行可视化
        :return 预计总耗时(不包括最后一个worker返回master的时延)
        """
        wk_tran, wk_cmpt = cls.plan2costs_chain(wk_elys, wk_cap, wk_bwth, lbsz, ly_comp)
        dp = cls.simulate_pipeline(wk_tran, wk_cmpt, nframe)
        if vis:
            cls.visualize_frames(wk_tran, wk_cmpt, dp)
        return dp[-1][-1]

    @classmethod
    def optimize_chain(cls, lb_wk_elys: List[List[int]], wk_cap: List[float], wk_bwth: List[float],
                       lbsz: List[float], ly_comp: List[float], nframe: int,
                       vis: bool = False, logger: logging.Logger = None) -> List[List[int]]:
        """对于链状的CNN，从负载最均衡的方案开始，优化nframe总耗时"""
        if logger is None:
            # 如果没有传入logger，则默认写入stdout
            logger = logging.getLogger('optimize_chain')
            if not logger.hasHandlers():
                # 因为这个logger是全局共用的，所以不能重复添加Handler
                logger.addHandler(logging.StreamHandler(sys.stdout))
                logger.setLevel(logging.DEBUG)
        wk_elys = lb_wk_elys
        cur_cost = cls.estimate_latency_chain(wk_elys, wk_cap, wk_bwth, lbsz, ly_comp, nframe, vis)
        logger.debug(f"wk_elys={wk_elys}, cost={cur_cost} (init)")
        wk_tran, _ = cls.plan2costs_chain(wk_elys, wk_cap, wk_bwth, lbsz, ly_comp)
        while True:
            mw = 1  # 传输耗时最大的worker，因为Worker0的传输耗时无法优化，所以从Worker1开始
            for w in range(2, len(wk_elys)):
                # w-1和w中至少一个有layer，不然没法优化
                if wk_tran[w] > wk_tran[mw] and len(wk_elys[w-1]) + len(wk_elys[w]) > 0:
                    mw = w
            pwk = mw-1  # mw左边第一个有layer的Worker
            while pwk >= 0 and len(wk_elys[pwk]) == 0:
                pwk -= 1
            ml = (wk_elys[pwk][-1] if pwk >= 0 else 0)  # mw数据传输对应的CNN层: 如果没找到说明是第0层
            elys = wk_elys[mw-1] + wk_elys[mw]
            bg_ly, ed_ly = elys[0], elys[-1]
            # 在mw-1和mw的任务中搜索割点，重新对mw-1和mw进行分配，要分配的层为elys，即[bg_ly, ed_ly]闭区间
            best_cost, best_wk_elys = cur_cost, wk_elys
            for l in elys:
                if lbsz[l] < lbsz[ml]:
                    tmp_wk_elys = wk_elys[:mw-1] + [list(range(bg_ly, l+1))] \
                                + [list(range(l+1, ed_ly+1))] + wk_elys[mw+1:]
                    tmp_cost = cls.estimate_latency_chain(tmp_wk_elys, wk_cap, wk_bwth, lbsz, ly_comp, nframe, False)
                    if tmp_cost < best_cost:
                        logger.debug(f"wk_elys={tmp_wk_elys}, cost={tmp_cost}")
                        best_cost, best_wk_elys = tmp_cost, tmp_wk_elys
                        if vis:  # 只对改进了的方案进行可视化
                            cls.estimate_latency_chain(tmp_wk_elys, wk_cap, wk_bwth, lbsz, ly_comp, nframe, True)
            if best_cost < cur_cost:
                cur_cost, wk_elys = best_cost, best_wk_elys
            else:
                break
        if vis:
            from matplotlib import pyplot as plt
            plt.show()  # 显示前后优化的多张图像
        return wk_elys

    @classmethod
    def visualize_frames(cls, wk_tran: List[float], wk_cmpt: List[float], fs_dp: List[List[float]]):
        """wk_tran[w], wk_cmpt[w]表示Worker w的传输耗时和计算耗时，先传输后计算
        fs_dp为 simulate_pipeline 的输出，不包括后面没有计算的Worker
        fs_dp[f][s]：第f帧第s个阶段完成时的耗时。s=2*w时表示w传输完成耗时，s=2*w+1时表示w计算完成耗时
        """
        from matplotlib import pyplot as plt
        import matplotlib.colors as mcolors
        act_nwk = len(fs_dp[0])//2
        nframe = len(fs_dp)
        fig = plt.figure()
        ax = fig.subplots()
        ax.invert_yaxis()
        ticklabels = ['m->w0', 'w0']
        for w in range(1, len(wk_cmpt)):
            ticklabels.extend([f'{w - 1}->{w}', f'w{w}'])
        plt.yticks(list(range(2 * len(wk_cmpt))), ticklabels)
        colors = list(mcolors.XKCD_COLORS.values())
        for f in range(nframe):
            for w in range(act_nwk):
                plt.barh(2 * w, wk_tran[w], left=fs_dp[f][2 * w] - wk_tran[w], color=colors[f])
                plt.barh(2 * w + 1, wk_cmpt[w], left=fs_dp[f][2 * w + 1] - wk_cmpt[w], color=colors[f])
            # 剩余的Worker画空的条形，使得纵轴显示出没有执行的worker
            for w in range(act_nwk, len(wk_cmpt)):
                plt.barh(2 * w, 0)
                plt.barh(2 * w + 1, 0)

    @classmethod
    def _predict_dag(cls, node_id: int, res_lcnz: List[List[float]],
                     dag: List[Node], predictors: List[Predictor]) -> None:
        """模仿core.raw_dnn.RawDNN.__execute_dag
        res_lcnz的每个元素必须初始化为空列表
        """
        if len(res_lcnz[node_id]) > 0:
            return
        acnz = []
        for aid in dag[node_id].ancients:
            if len(res_lcnz[aid]) > 0:
                acnz.append(res_lcnz[aid])
            else:
                return
        res_lcnz[node_id] = predictors[node_id].predict(acnz)
        for d in dag[node_id].descendants:
            cls._predict_dag(d, res_lcnz, dag, predictors)
