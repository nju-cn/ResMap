import logging
from collections import OrderedDict
from typing import Dict, Any

from torch import nn
from torchvision import models

from dnn_dag import make_dag
from dnn_layer import InputModule


def prepare_chain_model(model: nn.Module) -> Dict[str, Any]:
    """准备链状模型相关参数"""
    dnn_seq = OrderedDict({'ipt': InputModule()})  # DNN的各子模块按照顺序加入
    for name, child in model.named_children():
        dnn_seq[name] = child
    return {'dnn': nn.Sequential(dnn_seq), 'block_rules': {}, 'custom_dict': {}}


def prepare_alexnet() -> Dict[str, Any]:
    """准备AlexNet相关参数"""
    alexnet = models.alexnet(True)
    alexnet.eval()
    return prepare_chain_model(alexnet.features)


def prepare_vgg16() -> Dict[str, Any]:
    """准备VGG16相关参数"""
    vgg16 = models.vgg16(True)
    vgg16.eval()
    return prepare_chain_model(vgg16.features)


def prepare_vgg19() -> Dict[str, Any]:
    """准备VGG19相关参数"""
    vgg19 = models.vgg19(True)
    vgg19.eval()
    return prepare_chain_model(vgg19.features)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    dnn_args = prepare_vgg19()
    layers = make_dag(dnn_args['dnn'], dnn_args['block_rules'], logger)
    for ly in layers:
        print(ly)
