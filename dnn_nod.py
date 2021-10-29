from abc import abstractmethod, ABC

from dataclasses import dataclass, field
from typing import List, Tuple, Type

from torch import Tensor
from torch.nn import Module


@dataclass
class Nod:
    """用于生成Node，module可能只是一些计算结点的父结点，而不是单个计算结点"""
    id_: int
    module: Module  # module为DNN中的一个模块，不一定是计算结点，而calc一定是计算结点
    module_path: str  # module在整体DNN中的路径
    ds_nods: List['Nod']  # 后继结点
    ac_nods: List['Nod'] = field(default_factory=list)  # 前驱结点，默认为空列表

    def module_type(self):
        """本module所属的类名"""
        return type(self.module).__name__

    def __repr__(self):
        return f"Nod({self.id_}, {self.module_path}, {[d.id_ for d in self.ds_nods]})"

    def __hash__(self):
        """用作dict的key时，不同Nod用该对象的id区分"""
        return hash(id(self))


class InputModule(Module):
    """输入模块，作为DNN计算的唯一入口模块"""
    def forward(self, x: Tensor):
        return x


class ForkModule(Module):
    """Inception这种分叉结构的分叉模块。此类作为接口，继承此类用于标识DAG中一个分叉点
    若在子类中重写forward，则需要注意forward中不能对输入数据进行修改，因为在DNode中会保存输入数据"""
    @abstractmethod
    def forward(self, *inputs) -> Tensor:
        # inputs的类型与Module的默认类型保持了一致
        pass


class BasicFork(ForkModule):
    """输出=输入的分叉模块。因为很多分叉结构处理都只需要在前面加一个这样的分叉点，所以实现此类以便使用"""
    def forward(self, x: Tensor) -> Tensor:
        return x


class MergeModule(Module):
    """Inception这种分叉结构的汇聚模块。此类作为接口，继承此类用于标识DAG中一个汇聚点
    若在子类中重写forward，则需要注意forward中不能对输入数据进行修改，因为在DNode中会保存输入数据"""
    @abstractmethod
    def forward(self, *inputs) -> Tensor:
        # inputs的类型与Module的默认类型保持了一致
        pass


class BlockRule:
    """用于规定需要特殊处理的Module(称为Block)的子结点结构。非Sequential的Module一般都需要使用此类来规定计算过程"""
    @staticmethod
    @abstractmethod
    def is_target(module: Module) -> bool:
        """module是否应该由本规则处理"""
        pass

    @staticmethod
    @abstractmethod
    def build_dag(block: Module) -> List[Nod]:
        """给定block，构造出该Module到其子结点的Nod结构，返回所有Nod，并且按照id顺序排序
        返回列表中，首个必须为唯一的入口结点，末尾必须为唯一的出口结点
        返回的子结点不需要是叶子结点，系统会使用named_children对其子结点作进一步的展开，
        若某些自定义Module希望一次性执行而不被分成多个结点，则应重写(override)该Module的named_children函数
        block中若存在多分支，则分叉点的Module应继承ForkModule，汇聚点的Module应继承MergeModule，以便处理时识别与优化
        ForkModule和MergeModule对于分支的顺序要一致，即ForkModule各后继的顺序和MergeModule各前驱的顺序一致
        Nod.id_从0开始计数，各分支内id连续编号，分支之间编号顺序与分支顺序一致，汇聚点id大于前面所有结点的id
        Nod.module不能有inplace操作，这会影响Checker中ground-truth(detailed)的计算（因为计算这个的过程中保存了各Nod的数据）
        Nod.module_path为各模块名称简写，如ipt，conv1
        Nod.ac_nods与Nod.ds_nods按照数据流动方向填写"""
        pass


class OutRangeFactory(ABC):
    """out_range_factory的接口类，用于在CustomTypeInfo中定义Module的out_range函数
    因为out_range在网络传输时需要被序列化，而函数内定义的函数无法序列化，所以这里使用一个类来使得其可以被序列化"""
    def __init__(self, module: Module):
        """根据module进行初始化"""

    @abstractmethod
    def out_range(self, x1: int, x2: int, idx: int, strict: bool = False) -> Tuple[int, int]:
        """根据传入的module，给出输入在idx维度上的闭区间[x1,x2]时，module能够产生的在idx维度上相应的输出区间(闭区间)
        如果module对idx维度有padding操作，则[x1, x2]为padding后的输入区间
        idx有两种取值0和1，分别对应图像的高度(Tensor的第2维度)和图像的宽度(Tensor的第3维度)，这里的Tensor维度从0开始计
        当有[x1, x2]的数据无法算出任何结果时抛出异常NoOutException
        strict=True时为严格模式，不考虑边界条件，[x1,x2]可以为整体输入区间中的一部分，此时输出严格按照filter滑动的结果
        strict=False时，考虑边界条件，[x1,x2]为整体输入区间，x2为输入区间的右边界
        严格模式的out_range与严格模式的req_range是互为逆操作的：out_range(x1,x2)=(y1,y2)且req(y1,y2)=(x1,x2)
        """


class SimpleOutRangeFactory(OutRangeFactory):
    """最简单的一一映射，如ReLU"""
    def out_range(self, x1: int, x2: int, idx: int, strict: bool = False) -> Tuple[int, int]:
        return x1, x2


class ReqRangeFactory(ABC):
    """req_range_factory的接口类，用于在CustomTypeInfo中定义Module的req_range函数
    因为req_range在网络传输时需要被序列化，而函数内定义的函数无法序列化，所以这里使用一个类来使得其可以被序列化"""
    def __init__(self, module: Module):
        """根据module进行初始化"""

    @abstractmethod
    def req_range(self, x1: int, x2: int, idx: int, strict: bool = True) -> Tuple[int, int]:
        """根据传入的module，给出要输出在idx维度上的闭区间[x1,x2]，module需要的在idx维度上相应的输入区间(闭区间)
        如果module对idx维度有padding操作，则返回的输入区间应为padding后的输入区间
        idx有两种取值0和1，分别对应图像的高度(Tensor的第2维度)和图像的宽度(Tensor的第3维度)，这里的Tensor维度从0开始计
        strict=True时为严格模式，不考虑边界条件，[x1,x2]可以为整体输出区间中的一部分，此时将得到严格按照filter滑动的倒推结果
        strict=False时为宽松模式，考虑边界条件(如MaxPool2d的ceil_mode)，返回能输出[x1, x2]的最小输入范围
        严格模式的out_range与严格模式的req_range是互为逆操作的：out_range(x1,x2)=(y1,y2)且req(y1,y2)=(x1,x2)
        """


class SimpleReqRangeFactory(ReqRangeFactory):
    """最简单的一一映射，如ReLU"""
    def req_range(self, x1: int, x2: int, idx: int, strict: bool = True) -> Tuple[int, int]:
        return x1, x2


@dataclass
class CustomTypeInfo:
    """自定义module类的out_range和req_range信息，通常是在定义规则时新定义的类"""
    module_type: Type[Module]  # 自定义module所属的类
    out_range_factory: Type[OutRangeFactory]  # 该Module所属类生成out_range的方式
    req_range_factory: Type[ReqRangeFactory]  # 该Module所属类生成req_range的方式
