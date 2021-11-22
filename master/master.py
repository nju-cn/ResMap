import pickle
import time
from collections import deque
from typing import Tuple, Optional, Dict, Any
import threading

import cv2
import torch
from torch import Tensor
from torchvision.transforms import transforms

from core.dif_executor import DifJob
from core.raw_dnn import RawDNN
from core.util import cached_func, dnn_abbr
from rpc.msg_pb2 import ResultMsg, Req
from master.scheduler import Scheduler, SizedNode
from rpc.stub_factory import StubFactory
from worker.worker import IFR


class Master(threading.Thread):
    def __init__(self, stb_fct: StubFactory, config: Dict[str, Any]):
        super().__init__()
        self.__stb_fct = stb_fct
        raw_dnn = RawDNN(config['dnn_loader']())  # DNN相关的参数
        print("Profiling data sizes...")
        self.__frame_size = config['frame_size']
        dag = cached_func(f"{dnn_abbr(config['dnn_loader'])}.{self.__frame_size[0]}x{self.__frame_size[1]}.sz",
                          SizedNode.raw2dag_sized, raw_dnn, self.__frame_size)
        self.__raw_dnn: Optional[RawDNN] = (raw_dnn if config['check'] else None)
        self.__inputs: deque[Tuple[int, Tensor]] = deque()  # [(ifr_id, 输入数据)]
        self.__vid_cap = cv2.VideoCapture(config['video_path'])
        self.__wk_costs = {}
        for wid in config['addr']['worker'].keys():
            print(f"Getting layer costs from worker{wid}...")
            req = Req()
            wk_stb = self.__stb_fct.worker(wid)
            msg = wk_stb.profile_cost(req)
            self.__wk_costs[wid] = pickle.loads(msg.costs)
        print(f"Getting predictors from trainer...")
        predictors = self.__stb_fct.trainer().get_predictors(Req())
        self.__scheduler = Scheduler(dag, predictors)
        print("Master init finished")

    def run(self) -> None:
        ifr_cnt = 0
        pre_ipt = torch.zeros(self.__frame_size)
        while self.__vid_cap.isOpened() and ifr_cnt < 5:
            cur_ipt = self.get_ipt_from_video(self.__vid_cap, self.__frame_size)
            dif_ipt = cur_ipt - pre_ipt
            if self.__raw_dnn is not None:
                self.__inputs.append((ifr_cnt, cur_ipt))
            wk_jobs = self.__scheduler.gen_wk_jobs()
            wk_jobs[0].dif_job.id2dif = {0: dif_ipt}
            ifr = IFR(ifr_cnt, wk_jobs)
            print(f"send IFR{ifr.id}")
            self.__stb_fct.worker(ifr.wk_jobs[0].worker_id).new_ifr(ifr.to_msg())
            ifr_cnt += 1
            pre_ipt = cur_ipt
            time.sleep(5)

    def check_result(self, result_msg: ResultMsg) -> None:
        if self.__raw_dnn is not None:
            print(f"checking IFR{result_msg.ifr_id}")
            final_result = DifJob.arr3dmsg_tensor4d(result_msg.arr3d)
            assert result_msg.ifr_id == self.__inputs[0][0], "check error!"
            mci = self.__inputs.popleft()[1]
            results = self.__raw_dnn.execute(mci)
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
