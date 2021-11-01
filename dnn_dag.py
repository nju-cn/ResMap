import logging
import sys

import torch
from typing import List, Type, Dict, Set

from torch import Tensor
from torch.nn import Module

from dnn_layer import RawLayer, BlockRule, CustomTypeInfo
from echarts_util import gen_html
from node import Node
from lrd import out_range_factory, req_range_factory


def visualize_dag(layers: List[RawLayer], file_path: str) -> None:
    """使用echarts可视化生成的DAG图，写入dnn_layers.html"""
    e_nodes = [{"name": str(n.id_) + ',' + n.module_type(),
                "symbolSize": [len(str(n.id_) + ',' + n.module_type()) * 7, 20],  # 这里使用文字长度*7来控制矩形宽度，高度恒为20
                "tooltip": {"formatter": n.module_path}}  # 鼠标移动上去显示此模块的路径
               for n in layers]  # 用于向echarts传结点参数，包括结点名称和标识点大小（标识点形状已经在echarts_util中设置为矩形）
    e_links = []  # 用于向echarts传边的参数
    for layer in layers:
        for ds in layer.ds_layers:
            e_links.append({"source": layer.id_, "target": ds.id_})
    gen_html(e_nodes, e_links, file_path)


def _make_dag(dnn: Module, path: str, block_rules: Set[Type[BlockRule]], logger: logging.Logger) -> List[RawLayer]:
    """处理dnn，当前dnn对应的路径为path，生成相应的RawLayer(已彻底展开，均为叶子结点)，RawLayer.id_从0开始编号
    :return 所有RawLayer，按照id顺序排序，确保List中索引与RawLayer.id_相同"""
    if len(list(dnn.children())) <= 0:  # 没有子结点，直接返回
        return [RawLayer(0, dnn, path, [], [])]
    block_rule = None
    for br in block_rules:
        if br.is_target(dnn):
            block_rule = br
            break
    if block_rule is not None:  # dnn属于特殊处理的Block
        children = block_rule.build_dag(dnn)  # 直接子结点，可能一些子结点尚未彻底展开
    else:  # 不属于特殊处理的Block，则默认所有子结点为顺序执行
        children = []
        for name, child_mod in dnn.named_children():
            children.append(RawLayer(len(children), child_mod, name, [], []))  # 这里只填子结点缩写，与block_rule保持一致
        if len(children) > 1:  # 若不止一个，则要建立前后关系
            children[0].ds_layers = [children[1]]
            for i in range(1, len(children)-1):
                children[i].ac_layers = [children[i - 1]]
                children[i].ds_layers = [children[i + 1]]
            children[-1].ac_layers = [children[-2]]
    module_end = {}  # 父结点->展开得到的子结点中的出口结点（即子结点列表的末尾元素）
    layers = []
    for child in children:
        child.module_path = path + '.' + child.module_path  # 添加路径前缀
        chd_layers = _make_dag(child.module, child.module_path, block_rules, logger)
        module_end[child] = chd_layers[-1]
        if len(layers) > 0 and len(chd_layers) > 0:  # 前面有结点且当前存在子结点，chd_layers所有结点的id要向后平移，前后要建立连接
            offset = layers[-1].id_ + 1  # 偏移量
            for chd in chd_layers:
                chd.id_ += offset
            # 入口RawLayer继承child的前驱，因为前面的child已经不再使用，所以要使用module_end映射到原先child对应的末尾RawLayer
            chd_layers[0].ac_layers.extend([module_end[ac] for ac in child.ac_layers])
            for ac in child.ac_layers:
                module_end[ac].ds_layers.append(chd_layers[0])
        layers.extend(chd_layers)  # 添加进去
    return layers


def make_dag(dnn: Module, block_rules: Set[Type[BlockRule]], logger: logging.Logger = None) -> List[RawLayer]:
    """处理dnn，根据block_rules的规则，生成相应的由RawLayer构成的DAG图，输出使用logger
    :param dnn 要处理的dnn
    :param block_rules 特殊Module的处理规则
    :param logger 使用此logger输出相关信息"""
    if logger is None:
        # 如果没有传入logger，则默认写入stdout
        logger = logging.getLogger('make_dag')
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(logging.INFO)
    layers = _make_dag(dnn, 'dnn', block_rules, logger)
    for layer in layers:
        if isinstance(layer.module, torch.nn.ReLU):
            layer.module.inplace = False  # 确保各节点的输出数据是分开保存的
    visualize_dag(layers, "dag_layers.html")
    logger.info(f"The DAG of DNN has been generated, with {len(layers)} nodes, visualized in dag_layers.html")
    return layers


def dag_layer2node(dag_layers: List[RawLayer], custom_dict: Dict[Type[Module], CustomTypeInfo]) -> List[Node]:
    """将由RawLayer构成的DAG图转为由Node构成的DAG图
    :param dag_layers 原先的由RawLayer构成的DAG图
    :param custom_dict 自定义Module类对应的相关信息"""
    dag = []
    for layer in dag_layers:
        ancients = [d.id_ for d in layer.ac_layers]  # 前驱结点（按序排列）
        descendants = [d.id_ for d in layer.ds_layers]  # 后继结点（按序排列）
        if layer.module.__class__ in custom_dict:  # 自定义的模块，使用定义好的out和req
            custom_info = custom_dict[layer.module.__class__]
            out_range = custom_info.out_range_factory(layer.module).out_range
            req_range = custom_info.req_range_factory(layer.module).req_range
        else:  # 非自定义模块，使用factory计算out和req
            out_range = out_range_factory(layer.module)
            req_range = req_range_factory(layer.module)
        dag.append(Node(layer.id_, ancients, descendants, layer.module, out_range, req_range))
    return dag


def execute_dag(root: RawLayer, input_tensor: Tensor, results: List[Tensor]) -> List[Tensor]:
    """从root开始，以input_tensor为输入执行RawLayer组成的DAG，把各layer的计算结果放在results[root.id_]中
    results长度必须与总layer数相同"""
    if results[root.id_].nelement() > 0:  # 已经计算过，直接返回
        return results
    if len(root.ac_layers) <= 1:  # root为起始结点或链上结点，直接使用input_tensor计算
        with torch.no_grad():
            results[root.id_] = root.module(input_tensor)
    else:  # root有多个前驱，不使用input_tensor而使用results中的结果
        inputs = []
        for ac_layer in root.ac_layers:
            if results[ac_layer.id_].nelement() > 0:  # Tensor元素数>0，说明不为空，已计算出
                inputs.append(results[ac_layer.id_])  # 按顺序记录计算结果
            else:  # 这个前驱结点还没计算，root及其后继就先不计算，直接返回
                return results
        with torch.no_grad():
            results[root.id_] = root.module(*inputs)  # 将所有前驱结果作为输入
    for ds_layer in root.ds_layers:
        results = execute_dag(ds_layer, results[root.id_], results)
    return results
