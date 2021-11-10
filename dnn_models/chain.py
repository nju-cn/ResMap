import logging
from collections import OrderedDict

from torch import nn
from torchvision import models

from raw_dnn import RawDNN
from dnn_config import InputModule, DNNConfig


def prepare_chain_model(model: nn.Module) -> DNNConfig:
    """准备链状模型相关参数"""
    dnn_seq = OrderedDict({'ipt': InputModule()})  # DNN的各子模块按照顺序加入
    for name, child in model.named_children():
        dnn_seq[name] = child
    return DNNConfig(nn.Sequential(dnn_seq))


def prepare_alexnet() -> DNNConfig:
    """准备AlexNet相关参数"""
    alexnet = models.alexnet(True)
    alexnet.eval()
    return prepare_chain_model(alexnet.features)


def prepare_vgg16() -> DNNConfig:
    """准备VGG16相关参数"""
    vgg16 = models.vgg16(True)
    vgg16.eval()
    return prepare_chain_model(vgg16.features)


def prepare_vgg19() -> DNNConfig:
    """准备VGG19相关参数"""
    vgg19 = models.vgg19(True)
    vgg19.eval()
    return prepare_chain_model(vgg19.features)


if __name__ == '__main__':
    raw_dnn = RawDNN(prepare_vgg19())
    for ly in raw_dnn.layers:
        print(ly)
