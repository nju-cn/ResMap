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


class ExNode(Node):
    """保存实际的数据"""
    # noinspection PyMissingConstructor
    def __init__(self, node: Node) -> None:
        super().__dict__.update(node.__dict__)  # 使用Node的所有成员变量初始化DNode的所有成员变量
        self.__output: Optional[Tensor] = None  # 当前节点的输出数据，所有后继都完成时清空
        self.__finished: bool = False  # 当前节点是否已经完成

    def set_finish(self, output: Optional[Tensor]) -> None:
        """执行此函数的节点不运行execute，只是设置输入以便后继获取
        当output为None时，此节点不会被用到，只是将此节点标记为已完成，以便进行内存回收
        """
        assert self.__output is None and not self.__finished, "not-None or finished node cannot be set!"
        self.__output = output
        self.__finished = True

    def execute(self, *inputs: Tensor) -> None:
        """inputs为输入，执行并保存输出"""
        assert self.__output is None and not self.__finished, "output has been set!"
        with torch.no_grad():
            self.__output = self.calc(*inputs)
        self.__finished = True

    def get_output(self) -> Optional[Tensor]:
        return self.__output

    def finished(self) -> bool:
        """是否已完成"""
        return self.__finished

    def clear(self):
        """回收内存，但仍为finished状态"""
        self.__output = None

    def reset(self):
        """完全重置，回到初始状态"""
        self.clear()
        self.__finished = False


class Executor:
    """执行一次inference中的一组CNN层。喂进输入，得到输出"""
    def __init__(self, dnn_loader: Callable[[], Dict[str, Any]]):
        self.__logger = logging.getLogger(self.__class__.__name__)
        dnn_args = dnn_loader()  # DNN相关的参数
        self.__raw_layers = make_dag(dnn_args['dnn'], dnn_args['block_rules'], self.__logger)
        dag = dag_layer2node(self.__raw_layers, dnn_args['custom_dict'])
        self.__ex_dag = [ExNode(node) for node in dag]

    def exec(self, job: Job) -> Dict[int, Tensor]:
        """执行给定的Job，得到输出结果"""
        # print(f"Input Job:{job}")
        self.__init_job(job)
        # 执行job，获取输出
        for exec_id in job.exec_ids:
            print(f"exec layer{exec_id}")
            inputs = [self.__ex_dag[ds].get_output() for ds in self.__ex_dag[exec_id].ancients]
            self.__ex_dag[exec_id].execute(*inputs)
            # 内存回收
            for ac in self.__ex_dag[exec_id].ancients:
                if all(self.__ex_dag[ds].finished() for ds in self.__ex_dag[ac].descendants):
                    self.__ex_dag[ac].clear()
        out = {oid: self.__ex_dag[oid].get_output() for oid in job.out_ids}
        self.__clear_job(job)
        return out

    def __init_job(self, job: Job) -> None:
        """为job初始化：设置输入数据，并将输入节点的所有前驱标记为finished"""
        # 设置输入节点的数据
        for node_id, output in job.id2opt.items():
            self.__ex_dag[node_id].set_finish(output)
        # 标记前驱已完成。设置完再标记的目的是防止前面的输入节点被重复设置
        for node_id in job.id2opt.keys():
            self.__finish_ancients(node_id)

    def __finish_ancients(self, node_id: int) -> None:
        """递归将node_id的前驱标记成finished。node_id此时应该已经为finished"""
        for ac in self.__ex_dag[node_id].ancients:
            # 注意输入节点的前驱为Node.IPT_AC
            if ac != Node.IPT_AC and not self.__ex_dag[ac].finished():
                self.__ex_dag[ac].set_finish(None)
                self.__finish_ancients(ac)

    def __clear_job(self, job: Job) -> None:
        """清空这个Job相关的内存"""
        for node_id in job.id2opt.keys():
            self.__ex_dag[node_id].reset()
        for exec_id in job.exec_ids:
            self.__ex_dag[exec_id].reset()

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
    wk_jobs = [(0, Job(list(range(1, 55)), {0: ipt}, [49, 54])),
           (1, Job(list(range(55, 110)), {}, [101, 109])),
           (2, Job(list(range(110, 173)), {}, [172]))]
    raw_layers = executor.raw_layers()
    ex_out = {}
    for wk, cur_job in wk_jobs:
        id2opt = executor.exec(cur_job)
        if wk+1 < len(wk_jobs):
            wk_jobs[wk + 1][1].id2opt = id2opt
        else:
            ex_out = id2opt
    # 直接执行RawLayer，以检查正确性
    results = execute_dag(raw_layers[0], ipt, [Tensor() for _ in raw_layers])
    print(torch.max(torch.abs(results[-1]-ex_out[172])))
