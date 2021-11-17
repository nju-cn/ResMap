from typing import List, Tuple

from torch.nn import Module

from dnn_config import InputModule


class Node:
    def __init__(self, id_: int, ancients: List[int], descendants: List[int], calc: Module):
        self.id = id_
        if len(ancients) == 0:
            assert isinstance(calc, InputModule), f"{calc} is not InputModule but has no ancient!"
            self.ancients = []
        else:
            self.ancients = ancients  # 前驱结点，并按照分支处理顺序排序
        self.descendants = descendants  # 后继节点
        self.calc = calc  # 直接使用RawLayer中的Module


class SizedNode(Node):
    # noinspection PyMissingConstructor
    def __init__(self, node: Node, size: Tuple[int, int, int]):
        super().__dict__.update(node.__dict__)  # 使用Node的所有成员变量初始化这里的所有成员变量
        self.out_size: Tuple[int, int, int] = size  # (通道数, 行数, 列数)
