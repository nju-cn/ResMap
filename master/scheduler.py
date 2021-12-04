from typing import List, Optional, Tuple, Dict

import torch
from torch import Tensor

from core.dif_executor import DifJob
from core.executor import Node
from core.integral_executor import ExNode, IntegralExecutor, IntegralJob
from core.predictor import Predictor
from core.raw_dnn import RawDNN
from worker.worker import WkDifJob


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
        itg_extor = IntegralExecutor(raw_dnn, _SizingExNode)
        ipt = torch.rand(1, 3, *frame_size)
        job = IntegralJob(list(range(1, len(raw_dnn.layers))), [raw_dnn.layers[-1].id_], {0: ipt})
        itg_extor.exec(job)
        return [cls(se_node) for se_node in itg_extor.dag()]


class Scheduler:
    def __init__(self, dag: List[SizedNode], predictors: List[Predictor], wk_costs: List[List[float]]):
        self.__dag = dag
        self.__predictors = predictors
        base_wk = 0  # 编号最小的作为计算能力的baseline
        print(f"Worker{base_wk} is the baseline of computing capbility")
        self.__base_costs = wk_costs[base_wk]
        self.__wk_cap = []  # worker_id->相对计算能力
        for wk, costs in enumerate(wk_costs):
            assert costs[0] == 0, f"InputModule of Worker{wk} cost should be 0!"
            # Worker计算能力：除InputModule之外，各层耗时/base_wk该层耗时 的均值
            self.__wk_cap.append(sum(costs[l] / wk_costs[base_wk][l] for l in range(1, len(costs))) / (len(costs) - 1))
        print(f"wk_cap={self.__wk_cap}")
        print(f"base_costs={self.__base_costs}")
        wk_lynum = self.split_chain(self.__base_costs[1:], self.__wk_cap)  # 第0层不需要执行
        self.__lb_wk_layers = self.wk_lynum2layers(1, wk_lynum)
        print(f"load balance most: {self.__lb_wk_layers}")

    def gen_wk_jobs(self, dif_ipt: Tensor) -> List[WkDifJob]:
        cnz = [float(chan.count_nonzero()/chan.nelement()) for chan in dif_ipt[0]]
        lcnz = self.predict_dag(cnz, self.__dag, self.__predictors)
        lsz = self.lcnz2lsz(lcnz, self.__dag)
        est_latency = self.estimate_latency(self.__lb_wk_layers, [[lys[-1]] for lys in self.__lb_wk_layers],
                                            [sz*4 for sz in lsz], self.__wk_cap,
                                            [1024*1024] * len(self.__wk_cap), self.__base_costs, 3)
        print(f"est_latency={est_latency}s")
        return [WkDifJob(0, DifJob(list(range(1, 5)), [4], {0: dif_ipt})),
                WkDifJob(1, DifJob(list(range(5, 10)), [9], {})),
                WkDifJob(2, DifJob(list(range(10, 14)), [13], {}))]

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
    def wk_lynum2layers(cls, begin_layer: int, wk_lynum: List[int]) -> List[List[int]]:
        """根据各Worker执行层数，从begin_layer开始，按照执行顺序为Worker分配具体执行的层
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
    def estimate_latency(cls, wk_elys: List[List[int]], wk_olys: List[List[int]], lbsz: List[float],
                         wk_cap: List[float], wk_rate: List[float],
                         ly_comp: List[float], nframe: int) -> float:
        """输入计划任务和这个任务会执行的帧数，计算按照此计划任务执行的平均时延。注意：只考虑链状CNN！
        :param wk_elys 各worker执行哪些层
        :param wk_olys 各worker哪些层的输出数据需要传输给下一个worker
        :param lbsz 各层输出数据的大小，单位byte
        :param wk_cap worker的计算能力
        :param wk_rate wk_rate[w]为w-1与w之间的带宽。w=0时为master和worker0之间的带宽
        :param ly_comp 各layer的计算量
        :param nframe 总共会执行多少帧
        :return 预计总耗时(不包括最后一个worker返回master的时延)
        """
        # wk_trans[w]：单帧中，Worker w接收前一个Worker输出数据的传输耗时
        wk_trans = [lbsz[0] / wk_rate[0]] \
                   + [sum(lbsz[o] for o in wk_olys[w-1]) / wk_rate[w] for w in range(1, len(wk_cap))]
        # wk_cmpt[w]：单帧中，Worker w完成自己的所有层的计算耗时
        wk_cmpt = [sum(ly_comp[e] for e in wk_elys[w]) / wk_cap[w] for w in range(len(wk_cap))]
        # dp[f][w]：第f帧的第w个Worker完成耗时
        dp = [[0. for _ in range(len(wk_cap))] for _ in range(nframe)]
        dp[0][0] = wk_trans[0] + wk_cmpt[0]  # 传输+计算
        for w in range(1, len(wk_cap)):
            dp[0][w] = dp[0][w-1] + wk_trans[w] + wk_cmpt[w]
        for f in range(1, nframe):
            dp[f][0] = wk_trans[0] + wk_cmpt[0]
            for w in range(1, len(wk_cap)):
                dp[f][w] = max(dp[f-1][w], dp[f][w-1]+wk_trans[w]) + wk_cmpt[w]
        return dp[-1][-1]

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
