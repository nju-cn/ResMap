import logging
import time
from dataclasses import dataclass
from queue import Queue
from typing import Tuple, Optional, Dict, Any
import threading

import cv2
import torch
from torch import Tensor
from torchvision.transforms import transforms

from core.raw_dnn import RawDNN
from core.util import cached_func, dnn_abbr
from master.scheduler import Scheduler, SizedNode
from rpc.stub_factory import MStubFactory
from worker.worker import IFR


@dataclass
class PendingIpt:
    ifr_id: int
    ipt: Optional[Tensor]
    send_time: float


class Master(threading.Thread):
    def __init__(self, stb_fct: MStubFactory, config: Dict[str, Any]):
        super().__init__()
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__stb_fct = stb_fct
        self.__ifr_num = config['master']['ifr_num']
        self.__itv_time = config['master']['itv_time']
        self.__wk_num = len(config['port']['worker'])
        raw_dnn = RawDNN(config['dnn_loader']())  # DNN相关的参数
        self.__logger.info("Profiling data sizes...")
        self.__frame_size = config['frame_size']
        self.__raw_dnn: Optional[RawDNN] = (raw_dnn if config['check'] else None)
        self.__pd_que: Queue[PendingIpt] = Queue(config['master']['pd_num'])
        self.__begin_time = -1  # IFR0发出的时间
        self.__vid_cap = cv2.VideoCapture(config['video_path'])
        wk_costs = [[] for _ in range(self.__wk_num)]
        for wid in range(self.__wk_num):
            self.__logger.info(f"Getting layer costs from worker{wid}...")
            wk_costs[wid] = self.__stb_fct.worker(wid).layer_cost()
        self.__logger.info(f"Getting predictors from trainer...")
        predictors = self.__stb_fct.trainer().get_predictors()
        dag = cached_func(f"{dnn_abbr(config['dnn_loader'])}.{self.__frame_size[0]}x{self.__frame_size[1]}.sz",
                          SizedNode.raw2dag_sized, raw_dnn, self.__frame_size, logger=self.__logger)
        self.__scheduler = Scheduler(dag, predictors, wk_costs, config['master']['scheduler']['bandwidth'])
        self.__logger.info("Master init finished")

    def run(self) -> None:
        ifr_cnt = 0
        pre_ipt = torch.zeros(self.__frame_size)
        while self.__vid_cap.isOpened() and ifr_cnt < self.__ifr_num:
            cur_ipt = self.get_ipt_from_video(self.__vid_cap, self.__frame_size)
            dif_ipt = cur_ipt - pre_ipt
            wk_jobs = self.__scheduler.gen_wk_jobs(dif_ipt)
            ifr = IFR(ifr_cnt, wk_jobs)
            pd_data = (cur_ipt if self.__raw_dnn is not None else None)
            pd_ipt = PendingIpt(ifr.id, pd_data, -1)
            self.__pd_que.put(pd_ipt)  # 可能阻塞
            self.__logger.info(f"send IFR{ifr.id}")
            pd_ipt.send_time = time.time()
            self.__stb_fct.worker(ifr.wk_jobs[0].worker_id).new_ifr(ifr)
            if ifr_cnt == 0:
                self.__begin_time = pd_ipt.send_time
            ifr_cnt += 1
            pre_ipt = cur_ipt
            time.sleep(self.__itv_time)

    def report_finish(self, ifr_id: int, tensor: Tensor = None) -> None:
        pd_ipt = self.__pd_que.get()
        assert ifr_id == pd_ipt.ifr_id, "check error!"
        self.__logger.info(f"IFR{ifr_id} finished, latency={time.time()-pd_ipt.send_time}s")
        if ifr_id == self.__ifr_num - 1:  # 所有IFR均完成
            self.__logger.info(f"All {self.__ifr_num} IFRs finished, "
                               f"avg cost={(time.time()-self.__begin_time)/self.__ifr_num}s")
        if self.__raw_dnn is not None:
            assert tensor is not None, "check is True but result is None!"
            self.__logger.info(f"checking IFR{ifr_id}")
            results = self.__raw_dnn.execute(pd_ipt.ipt)
            self.__logger.info(f"IFR{ifr_id} error={torch.max(torch.abs(tensor-results[-1]))}")

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
