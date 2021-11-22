from typing import List

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch import Tensor
from torch.nn import Sequential, functional, Module, ReLU
from torchvision import models
from torchvision.models.resnet import Bottleneck

from core.predictor import Predictor, MLPPredictor, MLPsPredictor
from core.raw_dnn import RawDNN
from core.dnn_config import InputModule, MergeModule, BlockRule, RawLayer, BasicFork, DNNConfig


class BottleneckAdd(MergeModule):
    """Bottleneck末尾合并各分支的模块"""
    def forward(self, *inputs: Tensor) -> Tensor:
        return functional.relu(inputs[0]+inputs[1])


class BAPredictor(Predictor):
    """BottleneckAdd"""
    def __init__(self, module: torch.nn.Module):
        super().__init__(module)
        self.regrs: List[LinearRegression] = []

    def fit(self, afcnz: List[List[List[float]]], fcnz: List[List[float]]) -> 'BAPredictor':
        assert all(len(a_fncz) == len(fcnz) for a_fncz in afcnz)
        nframe, nchan = len(fcnz), len(fcnz[0])
        self.regrs = [LinearRegression() for _ in range(nchan)]
        for c, regr in enumerate(self.regrs):
            X = [[afcnz[a][f][c] for a in range(len(afcnz))] for f in range(1, nframe)]
            X = np.array(X)
            y = np.array([fcnz[f][c] for f in range(1, nframe)])
            regr.fit(X, y)
        return self

    def predict(self, acnz: List[List[float]]) -> List[float]:
        out = []
        for c, regr in enumerate(self.regrs):
            x = np.array([acnz[a][c] for a in range(len(acnz))])
            out.append(float(regr.predict([x])[0]))
        return out


class ConvPredictor(Predictor):
    def __init__(self, module: torch.nn.Module):
        super().__init__(module)
        assert isinstance(module, torch.nn.Conv2d)
        self.pedor = (MLPPredictor(module) if module.out_channels < 1000 else MLPsPredictor(module))

    def fit(self, afcnz: List[List[List[float]]], fcnz: List[List[float]]) -> 'ConvPredictor':
        self.pedor.fit(afcnz, fcnz)
        return self

    def predict(self, acnz: List[List[float]]) -> List[float]:
        return self.pedor.predict(acnz)


class BottleneckRule(BlockRule):

    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, Bottleneck)

    @staticmethod
    def build_dag(block: Bottleneck) -> List[RawLayer]:
        ipt = RawLayer(0, BasicFork(), 'ipt', [])
        conv1 = RawLayer(1, block.conv1, 'conv1', [], [ipt])
        bn1 = RawLayer(2, block.bn1, 'bn1', [], [conv1])
        relu1 = RawLayer(3, ReLU(inplace=False), 'relu1', [], [bn1])
        conv2 = RawLayer(4, block.conv2, 'conv2', [], [relu1])
        bn2 = RawLayer(5, block.bn2, 'bn2', [], [conv2])
        relu2 = RawLayer(6, ReLU(inplace=False), 'relu2', [], [bn2])
        conv3 = RawLayer(7, block.conv3, 'conv3', [], [relu2])
        bn3 = RawLayer(8, block.bn3, 'bn3', [], [conv3])
        ipt.ds_layers = [conv1]
        conv1.ds_layers = [bn1]
        bn1.ds_layers = [relu1]
        relu1.ds_layers = [conv2]
        conv2.ds_layers = [bn2]
        bn2.ds_layers = [relu2]
        relu2.ds_layers = [conv3]
        conv3.ds_layers = [bn3]
        if block.downsample is not None:  # 连接ipt的后继和merge的前驱时，要注意两者的分支顺序相同
            downsample = RawLayer(9, block.downsample, 'downsample', [], [ipt])
            merge = RawLayer(10, BottleneckAdd(), 'ba', [], [bn3, downsample])
            ipt.ds_layers.append(downsample)
            bn3.ds_layers = [merge]
            return [ipt, conv1, bn1, relu1, conv2, bn2, relu2, conv3, bn3, downsample, merge]
        else:
            merge = RawLayer(9, BottleneckAdd(), 'ba', [], [bn3, ipt])
            ipt.ds_layers.append(merge)
            bn3.ds_layers = [merge]
            return [ipt, conv1, bn1, relu1, conv2, bn2, relu2, conv3, bn3, merge]


def prepare_resnet50() -> DNNConfig:
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
    return DNNConfig(dnn, {BottleneckRule}, {BottleneckAdd: BAPredictor, torch.nn.Conv2d: ConvPredictor})


if __name__ == '__main__':
    raw_dnn = RawDNN(prepare_resnet50())
    for ly in raw_dnn.layers:
        print(ly)
