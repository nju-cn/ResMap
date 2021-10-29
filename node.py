import copy

from typing import List, Tuple, Callable, Optional, Dict
from torch.nn import Module, Conv2d, ReLU

from dnn_nod import InputModule
from lrd import out_range_factory, req_range_factory


# 各个worker自己保存，没有全局信息
class Node:

    IPT_AC = -1  # input ancient，为了方便统一处理，输入结点的前驱node_id设置为-1

    def __init__(self, id_: int, ancients: List[int], descendants: List[int], calc: Module,
                 out_range: Optional[Callable[[int, int, int, bool], Tuple[int, int]]] = None,
                 req_range: Optional[Callable[[int, int, int], Tuple[int, int]]] = None) -> None:
        """out_range和req_range不填则会用factory函数计算已定义的模块，自定义模块必须填，因为factory中没定义"""
        self.id = id_
        if len(ancients) == 0:
            assert isinstance(calc, InputModule), f"非入口模块{calc}没有前驱结点！"
            self.ancients = [self.IPT_AC]
        else:
            self.ancients = ancients  # 前驱结点，并按照分支处理顺序排序
        self.descendants = descendants  # 后继节点
        self.calc = copy.deepcopy(calc)  # 要执行的计算任务，为了防止calc被更改，使用了deepcopy
        if hasattr(calc, "padding"):
            # padding=Tuple[上下高度pad,左右宽度pad]
            if isinstance(calc.padding, int):
                self.padding = (calc.padding, calc.padding)
            else:
                self.padding = calc.padding
            self.calc.padding = (0, 0)
        else:
            self.padding = (0, 0)
        if isinstance(calc, ReLU):  # 将ReLU的inplace置为False，以便各结点数据分开保存
            self.calc.inplace = False
        # out_range(i_begin, i_end, idx)为第idx维[i_begin, i_end]的输入范围对应的输出范围，均为闭区间
        if out_range is None:  # 没有传入calc的out_range，默认使用out_range_factory计算
            self.__out_range = out_range_factory(calc)  # 这里必须使用原先的calc
        else:  # 传入了calc的out_range，以传入的为准
            self.__out_range = out_range
        # req_range(o_begin, o_end, idx)为第idx维[o_begin, o_end]的输出范围需要的输入范围，均为闭区间
        if req_range is None:  # 没有传入calc的req_range，默认使用req_range_factory计算
            self.__req_range = req_range_factory(calc)  # 这里必须使用原先的calc
        else:  # 传入了calc的req_range，以传入的为准
            self.__req_range = req_range

    def calc_out_range(self, x1: int, x2: int, idx: int, strict: bool = False) -> Tuple[int, int]:
        """pad后的输入区间[x1,x2]对应的输出，其中[x1,x2]必须为pad后的区间
        strict=True时，[x1,x2]可以为整体输入区间中的一部分，此时将得到严格按照filter滑动的结果
        strict=False时，[x1,x2]必须为整体输入区间，否则x2会被误认为输入区间的右边界，从而得到错误的结果
        对calc对应的out_range的直接封装，strict默认与calc一致"""
        return self.__out_range(x1, x2, idx, strict)

    def calc_req_range(self, x1: int, x2: int, idx: int, strict: bool = True) -> Tuple[int, int]:
        """输出区间[x1,x2]对应的pad后的区间，其中返回值为pad后的区间
        strict=True时，[x1,x2]可以为整体输出区间中的一部分，此时将得到严格按照filter滑动的倒推结果
        strict=False时，[x1,x2]必须为整体输出区间，否则x2会被误认为输出区间的右边界，从而得到错误的结果
        对calc对应的req_range的直接封装，strict默认为严格模式。返回值为pad后的区间"""
        return self.__req_range(x1, x2, idx, strict)


class RNode(Node):
    """RuntimeNode，运行了具体任务的Node，记录了整体输入区间信息"""

    # noinspection PyMissingConstructor
    def __init__(self, node: Node):
        super().__dict__.update(node.__dict__)  # 使用Node的所有成员变量初始化DNode的所有成员变量
        self.__irange: Tuple[int, int] = None  # 此结点需要处理输入数据的整体范围，不包括padding，即直接为前驱结点输出的区间。padding将加在irange两边
        self.__orange: Tuple[int, int] = None  # 此结点计算得到输出数据的整体范围，与padding无关

    def set_irange(self, i_begin: int, i_end: int) -> Tuple[int, int]:
        """设置此结点需要处理的原始输入数据范围为[i_begin, i_end](不包括padding)，并据此计算并设置相应输出数据范围
        :return 此输入对应的输出数据范围self.__orange"""
        # 若RNode存在多个前驱时，set_irange可能会被多次调用，但是先后设置的值必定相同
        assert self.__irange is None or self.__irange == (i_begin, i_end), f"Node{self.id}的irange被重复设置且先后设置值不同！"
        self.__irange = (i_begin, i_end)  # irange为原始输入数据区间，不包括padding
        # 必须先设置irange才能pad_range，因为pad_range需要irange
        pi_begin, pi_end = self.pad_irange(i_begin, i_end)  # pad后的输入区间
        self.__orange = super().calc_out_range(pi_begin, pi_end, 0)  # 输入包括padding，按照calc的ceil_mode情况计算输出区间
        return self.__orange

    def pad_irange(self, i_begin: int, i_end: int) -> Tuple[int, int]:
        """给出输入区间[i_begin, i_end]按照本结点padding之后对应的输入数据区间，包括可能存在的padding，即区间长度可能会变
        当输入区间在整体区间的两边时，包括padding；当输入区间在整体区间的中间时，不存在padding故不包括padding"""
        # 因为此函数在set_irange中会被用到，所以这里不能检查[i_begin,i_end]与整体输入区间的包含关系
        if i_begin == self.__irange[0]:  # 输入的是从起始点开始的数据，padding后仍然从起始点开始，因为包括了padding本身
            pi_begin = self.__irange[0]  # padded input begin，padding后数据区间起始点
        else:  # 起始点之后的数据，直接向后平移
            pi_begin = i_begin + self.padding[0]  # padded input begin，padding后数据区间起始点
        if i_end == self.__irange[1]:  # 输入的是以终止点结尾的数据，padding后向右平移了两倍的padding量，因为包括了末尾的padding
            pi_end = i_end + 2*self.padding[0]  # padded input end，padding后数据区间终止点
        else:  # 终止点之前的数据，直接向后平移
            pi_end = i_end + self.padding[0]
        return pi_begin, pi_end

    def unpad_irange(self, pi_begin: int, pi_end: int) -> Tuple[int, int]:
        """给出本结点pad后的输入区间[o_begin, o_end]对应的原始数据输入区间，返回的区间去掉了padding"""
        wpi_begin, wpi_end = self.pad_irange(self.__irange[0], self.__irange[1])  # whole padded input，pad后的整体输入区间
        # pad后的区间[pi_begin, pi_end]应当在整体pad后的区间内[wpi_begin, wpi_end]
        assert pi_begin >= wpi_begin, f"[{pi_begin}, {pi_end}] is out of [{wpi_begin}, {wpi_end}]!"
        if pi_begin == wpi_begin:  # pad后区间起始点等于整体数据起始点，pad前区间起始点就是原始输入区间irange起始点
            i_begin = self.__irange[0]
        else:  # pad后区间起始点在整体数据起始点之后，pad前区间起始点就要向前平移padding长度
            i_begin = pi_begin - self.padding[0]
        # pad后的区间[pi_begin, pi_end]应当在整体pad后的区间内[wpi_begin, wpi_end]
        assert pi_end <= wpi_end, f"[{pi_begin}, {pi_end}] is out of [{wpi_begin}, {wpi_end}]!"
        if pi_end == wpi_end:  # pad后区间终止点等于整体数据终止点，pad前区间终止点就是原始输入区间irange终止点
            i_end = self.__irange[1]
        else:  # pad后区间终止点在整体数据终止点之前，pad前区间终止点就要向前平移padding长度
            i_end = pi_end - self.padding[0]
        return i_begin, i_end

    def out_range(self, x1: int, x2: int, idx: int, pad: bool, fact: bool) -> Tuple[int, int]:
        """使用Node实际输入区间的信息，给出[x1, x2]对应的输出范围，idx表示这是行索引(0)还是列索引(1)
        pad表示[x1,x2]是否需要进行pad操作，True则计算前先pad处理，False则不额外处理
        fact表示是否根据实际输入数据对边界进行特殊处理，使得计算结果与calc的实际输出一致，True则与calc一致，False则对边界不作特殊处理
        fact=True时与calc_out_range的strict=False不同，fact=False时与calc_out_range的strict=True一致
        """
        if pad:  # [x1,x2]是原始数据区间，没有pad，需要进行pad
            px1, px2 = self.pad_irange(x1, x2)  # 加上padding
        else:  # [x1,x2]是pad后的区间，不需要pad
            px1, px2 = x1, x2
        # whole padded input，pad后的整体输入区间
        wpi_begin, wpi_end = self.pad_irange(self.__irange[0], self.__irange[1])
        assert wpi_begin <= px1 and px2 <= wpi_end, f"out_range of RNode{self.id}: " \
            f"[{px1}, {px2}] not in padded input range [{wpi_begin}, {wpi_end}]"
        if fact:  # fact=True，对边界进行特殊处理，对非边界不进行特殊操作
            if px2 == wpi_end:  # 区间在整体输入的右边界上，应使用strict=False来保持边界与实际输出一致
                return super().calc_out_range(px1, px2, idx, strict=False)
            else:  # 区间不在整体输入的右边界上，不需要处理边界，必须使用strict=True，否则会误将px2当做边界进行特殊处理
                return super().calc_out_range(px1, px2, idx, strict=True)
        else:  # fact=False，不进行任何特殊处理，故strict=True
            return super().calc_out_range(px1, px2, idx, strict=True)

    def req_range(self, x1: int, x2: int, idx: int, pad: bool, fact: bool) -> Tuple[int, int]:
        """使用Node实际输入区间的信息，给出[x1, x2]对应的输入范围，idx表示这是行索引(0)还是列索引(1)
        pad表示返回值是否包括pad，True则包括pad，False则不包括pad
        fact表示是否根据实际输入数据对边界进行特殊处理，使得计算结果与calc的实际输入一致，True则与calc一致，False则对边界不作特殊处理
        fact=True会考虑两种情况：Conv2d和MaxPool2d，
            Conv2d：严格按照filter滑动时边界上一些数据可能不会被用到，但实际上输入数据是包括了它们的，
            所以计算req时应该包括这部分数据，从而与输入数据保持一致
            MaxPool2d：当ceil_mode=True时，即使边界上不够一个kernel_size的大小，也可以忽略缺失的部分计算出结果，
            所以计算req时应该去掉多出来的这部分数据，从而与输入数据保持一致
        fact=True时与calc_req_range的strict=False不同，fact=False时与calc_req_range的strict=True一致
        """
        # 首先计算出严格模式下对应的pad后输入区间[py1, py2]，之后这两个值会根据fact取值情况被修改为实际对应的pad后输入区间
        py1, py2 = super().calc_req_range(x1, x2, idx, strict=True)  # padded y
        if fact:  # fact=True，对边界进行特殊处理，对非边界不进行特殊操作
            # whole padded input，pad后的整体输入区间
            wpi_begin, wpi_end = self.pad_irange(self.__irange[0], self.__irange[1])
            # 对于Conv2d的数据而言，py2与wpi_end差距在kernel_size之内，就算作[py1,py2]是在右边界上的数据了
            if isinstance(self.calc, Conv2d):  # Conv2d时右侧的数据可能会被忽略
                # Conv2d的输入必须在整体输入区间之内
                assert wpi_begin <= py1 and py2 <= wpi_end, f"req_range of RNode{self.id}: " \
                    f"[{py1}, {py2}] not in padded input range [{wpi_begin}, {wpi_end}]"
                # 若右侧范围小于kernel大小，则应包括右侧部分；py2=wpi_end时下面的赋值有没有都可以，因为不是特殊情况所以从if中排除了
                if 0 < wpi_end - py2 < self.calc.kernel_size[idx]:
                    py2 = wpi_end  # 右侧修改为输入区间右边界
            elif py2 > wpi_end:  # 对于其他类型的数据，严格模式倒推出来的范围可能超出实际输入范围，如MaxPool2d
                # 使用宽松模式计算要得到[x1,x2]所需要的最小区间[nsy1,nsy2]，用于检查正确性
                nsy1, nsy2 = super().calc_req_range(x1, x2, idx, strict=False)  # non-strict y
                # print(f"Node{self.id}使用了宽松模式，原先为[{py1}, {py2}]，现在为[{nsy1}, {wpi_end}] (pad=True)")
                # 宽松(non-strict)模式下，[nsy1,nsy2]应当在pad后的irange内，否则[nsy1,nsy2]无法计算得到[x1,x2]
                assert wpi_begin <= nsy1 and nsy2 <= wpi_end, f"req_range(non-strict) of RNode{self.id}: " \
                    f"[{nsy1}, {nsy2}] not in padded input range [{wpi_begin}, {wpi_end}]"
                py2 = wpi_end  # 右侧修改为输入区间右边界，即[py1,py2]与[wpi_begin,wpi_end]取交集
            # 若[py1,py2]不在Conv2d的右边界，且在完整输入区间内，则[py1,py2]不需要修改
        if pad:  # 直接返回pad后的输入区间[py1, py2]
            return py1, py2
        else:  # 返回pad后区间[py1,py2]对应的原始输入区间，此区间包括了padding
            return self.unpad_irange(py1, py2)

    def get_irange(self) -> Tuple[int, int]:
        return self.__irange

    def get_orange(self) -> Tuple[int, int]:
        return self.__orange

    @classmethod
    def req_range_through(cls, cur_node_id: int, pre_node_id: int, o_begin: int, o_end: int,
                          node_orange: Dict[int, Tuple[int, int]], r_dag: List['RNode']) \
            -> Optional[Tuple[int, int]]:
        """从后向前倒推，cur_node_id的输出区间[o_begin, o_end](无pad)需要的pre_node_id的输入区间(有pad)
        pre_node_id应当与cur_node_id存在前驱关系，即存在一条路径沿着数据流动的方向从pre_node_id到cur_node_id
        向node_orange写入沿途所有Node相应的输出范围(无pad)
        返回None说明沿着数据流动的方向不存在从pre_node_id到cur_node_id的路径
        """
        # print(f"req: {cur_node_id}->{pre_node_id}: [{o_begin}, {o_end}]")
        if cur_node_id in node_orange:
            no_begin, no_end = node_orange[cur_node_id]  # node_orange begin/end
            if no_begin <= o_begin and o_end <= no_end:  # 如果所求[o_begin,o_end]已在[no_begin,no_end]内就直接返回
                return None
            else:  # 只要所求区间不完全在[no_begin, no_end]内，就要重新向前计算
                o_begin, o_end = min(no_begin, o_begin), max(no_end, o_end)  # 对两区间取并集，对前面重新计算
        # print(f"Node{cur_node_id}: [{o_begin}, {o_end}]")
        node_orange[cur_node_id] = (o_begin, o_end)
        if cur_node_id == Node.IPT_AC:  # 倒推到了最前面，说明不存在前驱关系，返回None
            return None
        if cur_node_id == pre_node_id:  # 同一个结点
            return r_dag[cur_node_id].req_range(o_begin, o_end, 0, pad=True, fact=True)
        i_begin, i_end = r_dag[cur_node_id].req_range(o_begin, o_end, 0, pad=False, fact=True)  # 无pad的输入，即前驱的输出区间
        merge_orange = None  # 合并Node
        for ac in r_dag[cur_node_id].ancients:
            rg = cls.req_range_through(ac, pre_node_id, i_begin, i_end, node_orange, r_dag)
            if rg is not None:  # 不同分支计算同一输出区间需要的输入区间可能不同，所以各分支都要计算，取并集
                if merge_orange is None:
                    merge_orange = rg
                else:  # 取并集：找最小的满足所有分支的区间
                    merge_orange = min(merge_orange[0], rg[0]), max(merge_orange[1], rg[1])
                # print(f"Merge{cur_node_id}-Node{ac}: rg={rg}, merge_orange={merge_orange}")
        return merge_orange

    @classmethod
    def out_range_through(cls, cur_node_id: int, nxt_node_id: int, pi_begin: int, pi_end: int,
                          node_orange: Dict[int, Tuple[int, int]], r_dag: List['RNode']) \
            -> Optional[Tuple[int, int]]:
        """cur_node_id输入区间(有pad)为[pi_begin, pi_end]，沿着所有路径计算到nxt_node_id
        沿途所有Node可以得到的输出区间填写在node_orange中
        返回nxt_node_id的输出区间"""
        o_begin, o_end = r_dag[cur_node_id].out_range(pi_begin, pi_end, 0, pad=False, fact=True)
        if cur_node_id in node_orange:  # 已访问过，返回None
            no_begin, no_end = node_orange[cur_node_id]  # node_orange begin/end
            # 若[no_begin,no_end]已在[o_begin,o_end]内，则合并结点[no_begin,no_end]必定可以由当前区间算出来，而且多出来的部分也必定算不出来
            if o_begin <= no_begin and no_end <= o_end:  # 多出来的部分算不出来，所以返回None
                return None
            else:  # 不完全在里面，则当前这个区间可能会导致合并结点输出区间变得更小，所以要重新计算
                o_begin, o_end = max(no_begin, o_begin), min(no_end, o_end)  # 对两区间取交集，对后面重新计算
        node_orange[cur_node_id] = o_begin, o_end
        if cur_node_id == nxt_node_id:  # 同一个结点
            return node_orange[cur_node_id]
        fork_orange = None  # 分叉Node
        # print(f"{cur_node_id}后继：{r_dag[cur_node_id].descendants}")
        for ds in r_dag[cur_node_id].descendants:
            # print(f"out: {cur_node_id}->{nxt_node_id}: {ds}")
            npi_begin, npi_end = r_dag[ds].pad_irange(o_begin, o_end)  # next padded input begin/end
            # rg为从ds这个分支可以得到的nxt_node_id的输出区间
            rg = cls.out_range_through(ds, nxt_node_id, npi_begin, npi_end, node_orange, r_dag)
            if rg is not None:
                if fork_orange is None:
                    fork_orange = rg
                else:  # 从各分支计算可以得到的最终输出区间取交集
                    fork_orange = max(fork_orange[0], rg[0]), min(fork_orange[1], rg[1])
                # print(f"out: Merge{cur_node_id}-Node{ds}: rg={rg}, fork_orange={fork_orange}")
        return fork_orange
