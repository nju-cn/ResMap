import logging
from typing import Union, Tuple, List

from torch import Tensor
from torch.nn import Sequential, functional, Module, ReLU
from torchvision import models
from torchvision.models.resnet import Bottleneck

from dnn_dag import make_dag
from dnn_nod import InputModule, CustomTypeInfo, MergeModule, BlockRule, Nod, \
    BasicFork, SimpleOutRangeFactory, SimpleReqRangeFactory


class BottleneckAdd(MergeModule):
    """Bottleneck末尾合并各分支的模块"""
    def forward(self, inputs: Union[Tuple[Tensor, ...], List[Tensor]]) -> Tensor:
        return functional.relu(inputs[0]+inputs[1])


class BottleneckRule(BlockRule):

    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, Bottleneck)

    @staticmethod
    def build_dag(block: Bottleneck) -> List[Nod]:
        ipt = Nod(0, BasicFork(), 'ipt', [])
        conv1 = Nod(1, block.conv1, 'conv1', [], [ipt])
        bn1 = Nod(2, block.bn1, 'bn1', [], [conv1])
        relu1 = Nod(3, ReLU(inplace=False), 'relu1', [], [bn1])
        conv2 = Nod(4, block.conv2, 'conv2', [], [relu1])
        bn2 = Nod(5, block.bn2, 'bn2', [], [conv2])
        relu2 = Nod(6, ReLU(inplace=False), 'relu2', [], [bn2])
        conv3 = Nod(7, block.conv3, 'conv3', [], [relu2])
        bn3 = Nod(8, block.bn3, 'bn3', [], [conv3])
        ipt.ds_nods = [conv1]
        conv1.ds_nods = [bn1]
        bn1.ds_nods = [relu1]
        relu1.ds_nods = [conv2]
        conv2.ds_nods = [bn2]
        bn2.ds_nods = [relu2]
        relu2.ds_nods = [conv3]
        conv3.ds_nods = [bn3]
        if block.downsample is not None:  # 连接ipt的后继和merge的前驱时，要注意两者的分支顺序相同
            downsample = Nod(9, block.downsample, 'downsample', [], [ipt])
            merge = Nod(10, BottleneckAdd(), 'ba', [], [bn3, downsample])
            ipt.ds_nods.append(downsample)
            bn3.ds_nods = [merge]
            return [ipt, conv1, bn1, relu1, conv2, bn2, relu2, conv3, bn3, downsample, merge]
        else:
            merge = Nod(9, BottleneckAdd(), 'ba', [], [bn3, ipt])
            ipt.ds_nods.append(merge)
            bn3.ds_nods = [merge]
            return [ipt, conv1, bn1, relu1, conv2, bn2, relu2, conv3, bn3, merge]


def prepare_resnet50():
    resnet50 = models.resnet50(True)
    resnet50.eval()
    dnn = Sequential(
        InputModule(),
        resnet50.conv1,
        resnet50.bn1,
        resnet50.relu,
        resnet50.maxpool,
        resnet50.layer1,
        resnet50.layer2,
        resnet50.layer3,
        resnet50.layer4
    )
    ba_info = CustomTypeInfo(BottleneckAdd, SimpleOutRangeFactory, SimpleReqRangeFactory)
    custom_dict = {BottleneckAdd: ba_info}
    return {'dnn': dnn, 'block_rules': {BottleneckRule}, 'custom_dict': custom_dict}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    dnn_args = prepare_resnet50()
    nods = make_dag(dnn_args['dnn'], dnn_args['block_rules'], logger)
    # for nd in nods:
    #     print(nd)
