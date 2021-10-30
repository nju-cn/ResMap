import logging
from typing import Union, Tuple, List

import torch
from torch import Tensor
from torch.nn import Module, ReLU, Sequential
from torchvision import models
from torchvision.models.googlenet import Inception, BasicConv2d

from dnn_dag import make_dag
from dnn_layer import MergeModule, BlockRule, RawLayer, InputModule, CustomTypeInfo, \
    BasicFork, SimpleOutRangeFactory, SimpleReqRangeFactory


class InceptionCat(MergeModule):
    """Googlenet中Inception末尾合并各分支的模块"""
    def forward(self, *inputs: Tensor) -> Tensor:
        return torch.cat(inputs, 1)


class InceptionRule(BlockRule):

    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, Inception)

    @staticmethod
    def build_dag(block: Inception) -> List[RawLayer]:
        ipt = RawLayer(0, BasicFork(), 'ipt', [], [])
        branch1 = RawLayer(1, block.branch1, 'branch1', [], [ipt])
        branch2 = RawLayer(2, block.branch2, 'branch2', [], [ipt])
        branch3 = RawLayer(3, block.branch3, 'branch3', [], [ipt])
        branch4 = RawLayer(4, block.branch4, 'branch4', [], [ipt])
        ipt.ds_layers = [branch1, branch2, branch3, branch4]
        ic = RawLayer(5, InceptionCat(), 'ic', [], [branch1, branch2, branch3, branch4])
        branch1.ds_layers = [ic]
        branch2.ds_layers = [ic]
        branch3.ds_layers = [ic]
        branch4.ds_layers = [ic]
        return [ipt, branch1, branch2, branch3, branch4, ic]


class BasicConv2dRule(BlockRule):
    """GoogLeNet中BasicConv2d末尾进行ReLU操作"""
    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, BasicConv2d)

    @staticmethod
    def build_dag(block: BasicConv2d) -> List[RawLayer]:
        # 这里是一个完整的顺序操作，所以不需要ForkModule和MergeModule
        conv = RawLayer(0, block.conv, 'conv', [], [])
        bn = RawLayer(1, block.bn, 'bn', [], [conv])
        relu = RawLayer(2, ReLU(inplace=False), 'relu', [], [bn])
        conv.ds_layers = [bn]
        bn.ds_layers = [relu]
        return [conv, bn, relu]


def prepare_googlenet():
    # 准备模型
    googlenet = models.googlenet(True)
    googlenet.eval()
    # 准备DAG
    dnn = Sequential(
        InputModule(),
        googlenet.conv1,
        googlenet.maxpool1,
        googlenet.conv2,
        googlenet.conv3,
        googlenet.maxpool2,
        googlenet.inception3a,
        googlenet.inception3b,
        googlenet.maxpool3,
        googlenet.inception4a,
        googlenet.inception4b,
        googlenet.inception4c,
        googlenet.inception4d,
        googlenet.inception4e,
        googlenet.maxpool4,
        googlenet.inception5a,
        googlenet.inception5b
    )
    ic_info = CustomTypeInfo(InceptionCat, SimpleOutRangeFactory, SimpleReqRangeFactory)
    custom_dict = {InceptionCat: ic_info}
    return {'dnn': dnn, 'block_rules': {InceptionRule, BasicConv2dRule}, 'custom_dict': custom_dict}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    dnn_args = prepare_googlenet()
    layers = make_dag(dnn_args['dnn'], dnn_args['block_rules'], logger)
    for nd in layers:
        print(nd)
