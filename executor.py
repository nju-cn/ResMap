import logging
from dataclasses import dataclass
from typing import List, Dict, Callable, Any, Tuple, Optional

import cv2
import torch
from torch import Tensor
from torchvision.transforms import transforms

from dnn_dag import make_dag, dag_layer2node, execute_dag
from node import Node, RNode
from dnn_models.chain import prepare_alexnet, prepare_vgg19
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50


@dataclass
class Job:
    exec_ids: List[int]  # 要执行的这组CNN层的id，按照执行顺序排列
    id2opt: Dict[int, Tensor]  # 先前完成的Job得到的输出，node_id->Tensor
    out_ids: List[int]  # 这组CNN层中输出层的id


@dataclass
class IFR:
    """一个帧对应的Worker执行计划，在Worker之间按照顺序传递
    当一帧到来时，Master生成每个Worker相应Job的exec_ids和out_ids
    每次传递时，从中删除已完成的Job，设置下一个Job的id2opt
    """
    wk_jobs: List[Tuple[int, Job]]  # 所有Worker按照执行顺序排列，[(worker_id, 要执行的Job)]


class ExNode(Node):
    """保存实际的数据"""
    # noinspection PyMissingConstructor
    def __init__(self, node: Node):
        super().__dict__.update(node.__dict__)  # 使用Node的所有成员变量初始化DNode的所有成员变量
        self.__output: Optional[Tensor] = None

    def execute(self, *inputs: Tensor):
        with torch.no_grad():
            self.__output = self.calc(*inputs)

    def set_output(self, output: Tensor):
        assert self.__output is None, "not-None output cannot be set!"
        self.__output = output

    def get_output(self) -> Optional[Tensor]:
        return self.__output

    def clear(self):
        self.__output = None


class Executor:
    """执行一次inference中的一组CNN层。喂进输入，得到输出"""
    def __init__(self, dnn_loader: Callable[[], Dict[str, Any]]):
        self.__logger = logging.getLogger(self.__class__.__name__)
        dnn_args = dnn_loader()  # DNN相关的参数
        raw_layers = make_dag(dnn_args['dnn'], dnn_args['block_rules'], self.__logger)
        self.__raw_layers = raw_layers
        dag = dag_layer2node(raw_layers, dnn_args['custom_dict'])
        self.__data_dag = [ExNode(node) for node in dag]

    def exec(self, job: Job) -> Dict[int, Tensor]:
        """执行给定的Job，得到输出结果"""
        # 清空原先数据，设置最新数据
        for e_node in self.__data_dag:
            e_node.clear()
        for node_id, output in job.id2opt.items():
            self.__data_dag[node_id].set_output(output)
        # 执行job，获取输出
        for exec_id in job.exec_ids:
            print(f"exec layer{exec_id}")
            inputs = [self.__data_dag[ds].get_output() for ds in self.__data_dag[exec_id].ancients]
            self.__data_dag[exec_id].execute(*inputs)
        out = {}
        for out_id in job.out_ids:
            out[out_id] = self.__data_dag[out_id].get_output()
        return out

    def raw_layers(self):
        return self.__raw_layers


def get_ipt_from_video(capture):
    ret, frame_bgr = capture.read()
    if not ret:
        print("failed to read")
        exit()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((270, 480)),
        transforms.ToTensor()
    ])
    input_batch = preprocess(frame_rgb)
    return input_batch.unsqueeze(0)


if __name__ == '__main__':
    executor = Executor(prepare_resnet50)
    cap = cv2.VideoCapture(f'test_scripts/media/树荫道路.mp4')
    ipt = get_ipt_from_video(cap)
    ifr = IFR([(0, Job(list(range(1, 55)), {0: ipt}, [49, 54])),
           (1, Job(list(range(55, 110)), {}, [101, 109])),
           (2, Job(list(range(110, 173)), {}, [172]))])
    raw_layers = executor.raw_layers()
    ex_out = {}
    for wk, cur_job in ifr.wk_jobs:
        id2opt = executor.exec(cur_job)
        if wk+1 < len(ifr.wk_jobs):
            ifr.wk_jobs[wk + 1][1].id2opt = id2opt
        else:
            ex_out = id2opt
    # 直接执行RawLayer，以检查正确性
    results = execute_dag(raw_layers[0], ipt, [Tensor() for _ in raw_layers])
    print(torch.max(torch.abs(results[-1]-ex_out[172])))
