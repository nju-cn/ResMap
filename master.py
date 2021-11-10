import pickle
import time
from collections import deque
from typing import Callable, Dict, Any, Tuple
import threading

import cv2
import torch
from torch import Tensor
from torchvision.transforms import transforms

from dif_executor import DifJob
from dnn_dag import make_dag, dag_layer2node, execute_dag
from msg_pb2 import IFRMsg, Arr3dMsg, ResultMsg
from node import RNode
from scheduler import Scheduler
from worker import IFR


class Master(threading.Thread):
    def __init__(self, dnn_loader: Callable[[], Dict[str, Any]],
                 video_path: str, frame_size: Tuple[int, int], check: bool,
                 send_ifr_async: Callable[[IFRMsg], None], config: Dict[str, Any]):
        super().__init__()
        dnn_args = dnn_loader()  # DNN相关的参数
        raw_layers = make_dag(dnn_args['dnn'], dnn_args['block_rules'])
        dag = dag_layer2node(raw_layers, dnn_args['custom_dict'])
        self.__r_dag = [RNode(node) for node in dag]
        RNode.init_rdag(self.__r_dag, 0, frame_size[1]-1)  # 注意：一般列数>行数，这里直接使用列数
        if check:
            self.__raw_layers = raw_layers
        else:
            self.__raw_layers = None
        self.__inputs: deque[Tuple[int, Tensor]] = deque()  # [(ifr_id, 输入数据)]
        self.__vid_cap = cv2.VideoCapture(video_path)
        self.__frame_size = frame_size
        self.__scheduler = Scheduler(self.__r_dag)
        self.__send_ifr_async = send_ifr_async

    def run(self) -> None:
        ifr_cnt = 0
        pre_ipt = torch.zeros(1, 3, *self.__frame_size)
        while self.__vid_cap.isOpened() and ifr_cnt < 5:
            cur_ipt = self.get_ipt_from_video(self.__vid_cap, self.__frame_size)
            dif_ipt = cur_ipt - pre_ipt
            if self.__raw_layers is not None:
                self.__inputs.append((ifr_cnt, cur_ipt))
            wk_jobs = self.__scheduler.gen_wk_jobs()
            wk_jobs[0].dif_job.id2dif = {0: dif_ipt}
            ifr = IFR(ifr_cnt, wk_jobs)
            print(f"send IFR{ifr.id}")
            self.__send_ifr_async(ifr.to_msg())
            ifr_cnt += 1
            pre_ipt = cur_ipt
            time.sleep(3)

    def check_result(self, result_msg: ResultMsg) -> None:
        if self.__raw_layers is not None:
            print(f"checking IFR{result_msg.ifr_id}")
            final_result = DifJob.arr3dmsg_tensor4d(result_msg.arr3d)
            assert result_msg.ifr_id == self.__inputs[0][0], "check error!"
            mci = self.__inputs.popleft()[1]
            # TODO: 这里得到的results[1:]中每个元素都有1e-5到1e-6级别的误差
            #   封装DNNConfig和RawDNN
            results = execute_dag(self.__raw_layers[0], mci,
                                  [Tensor() for _ in self.__raw_layers])
            print(f"IFR{result_msg.ifr_id} error={torch.max(torch.abs(final_result-results[-1]))}")

    @staticmethod
    def get_ipt_from_video(capture: cv2.VideoCapture, frame_size: Tuple[int, int]) -> Tensor:
        ret, frame_bgr = capture.read()
        if not ret:
            raise Exception("failed to read video")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(frame_size),
            transforms.ToTensor()
        ])
        ipt = preprocess(frame_rgb)
        return ipt.unsqueeze(0)
