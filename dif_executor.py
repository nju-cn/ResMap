import pickle
from dataclasses import dataclass
from typing import Dict, Callable, Any, List, Tuple

import cv2
import torch
from torch import Tensor
from scipy.sparse import csr_matrix

from integral_executor import IntegralExecutor, IntegralJob, Job, Executor
from dnn_models.resnet import prepare_resnet50
# from msg_pb2 import JobMsg


class InCache:
    def __init__(self) -> None:
        self.__id2opt = {}  # 上一帧的数据

    def update(self, id2dif: Dict[int, Tensor]) -> Dict[int, Tensor]:
        """更新内部数据并返回更新后的值，Dict中的key以id2dif为准
        若self.__id2opt中没有某key但id2dif中有，则 新的id2opt[该key]=dif
        若self.__id2opt中有某key但id2dif中没有，则 新的id2opt[该key]不存在
        """
        id2opt = {}
        for nid, dif in id2dif.items():
            if nid not in self.__id2opt:
                id2opt[nid] = dif
            else:
                id2opt[nid] = self.__id2opt[nid] + dif
        self.__id2opt = id2opt
        return self.__id2opt


class OutCache:
    def __init__(self) -> None:
        self.__id2opt = {}  # 上一帧的数据

    def diff(self, id2opt: Dict[int, Tensor]) -> Dict[int, Tensor]:
        """更新内部数据并返回 (最新数据-之前数据)
        若旧数据中没有某key但新数据中有，则 id2dif[该key]=新opt
        若旧数据中有某key但新数据中没有，则 id2dif[该key]不存在
        """
        id2dif = {}
        for nid, opt in id2opt.items():
            if nid not in self.__id2opt:
                id2dif[nid] = opt
            else:
                id2dif[nid] = opt - self.__id2opt[nid]
        self.__id2opt = id2opt
        return id2dif

    def get(self) -> Dict[int, Tensor]:
        return self.__id2opt


@dataclass
class DifJob(Job):
    id2dif: Dict[int, Tensor]  # dif为(后一帧-前一帧)，node_id->Tensor

    def __init__(self, exec_ids: List[int], out_ids: List[int], id2dif: Dict[int, Tensor]):
        super().__init__(exec_ids, out_ids)
        self.id2dif = id2dif

    # @staticmethod
    # def pk2tensor(pk3d: bytes) -> Tensor:
    #     csr2ds = pickle.loads(pk3d)
    #     tensor3ds = [torch.as_tensor(csr2d.toarray()).unsqueeze(0) for csr2d in csr2ds]
    #     return torch.cat(tensor3ds).unsqueeze(0)
    #
    # @staticmethod
    # def tensor2pk(tensor: Tensor) -> bytes:
    #     csr2ds = [csr_matrix(mtrx2d) for mtrx2d in tensor.numpy()[0]]
    #     return pickle.dumps(csr2ds)
    #
    # @classmethod
    # def from_msg(cls, job_msg: JobMsg) -> 'DifJob':
    #     id2dif = {nid: cls.pk2tensor(dif_pk) for nid, dif_pk in job_msg.id2dif.items()}
    #     return DifJob(job_msg.exec_ids, job_msg.out_ids, id2dif)
    #
    # def to_msg(self) -> JobMsg:
    #     id2dif = {nid: self.tensor2pk(dif) for nid, dif in self.id2dif.items()}
    #     return JobMsg(exec_ids=self.exec_ids, out_ids=self.out_ids, id2dif=id2dif)

    def to_integral(self, in_cache: InCache) -> IntegralJob:
        """借助InCache，把DifJob转成完整的IntegralJob，更新InCache"""
        return IntegralJob(self.exec_ids, self.out_ids, in_cache.update(self.id2dif))

    def clear(self) -> None:
        """清空自己的全部字段"""
        self.exec_ids.clear()
        self.id2dif.clear()
        self.out_ids.clear()


class DifExecutor(Executor):
    """内部缓存上次的执行结果，输入DifJob，得到输出
    DifJob必须为 这次数据-上次数据"""

    def __init__(self, dnn_loader: Callable[[], Dict[str, Any]]):
        self.__itg_extor = IntegralExecutor(dnn_loader)
        self.__in_cache = InCache()  # DifJob中上一帧输入的缓存，获得输入时更新
        self.__out_cache = OutCache()  # DifJob中上一帧输出的缓存，获得输出时更新

    def exec(self, dif_job: DifJob) -> Dict[int, Tensor]:
        """输入DifJob，输出Dif
        :param dif_job 输入节点的dif 这一帧-上一帧
        :return 输出节点的dif 这一帧-上一帧
        """
        job = dif_job.to_integral(self.__in_cache)
        id2opt = self.__itg_extor.exec(job)
        return self.__out_cache.diff(id2opt)

    def last_out(self) -> Dict[int, Tensor]:
        """获取最新一次运行的原始输出结果（不是Dif）"""
        return self.__out_cache.get()

    def check_exec(self, input_: Tensor) -> List[Tensor]:
        return self.__itg_extor.check_exec(input_)
