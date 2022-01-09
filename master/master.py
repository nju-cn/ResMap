import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Type, List
import threading

import cv2
import torch
from torch import Tensor
from torchvision.transforms import transforms

from core.executor import Job
from core.raw_dnn import RawDNN
from core.util import cached_func
from master.scheduler import SizedNode
from rpc.stub_factory import MStubFactory
from worker.worker import IFR


@dataclass
class PendingIpt:
    ifr_id: int
    ipt: Optional[Tensor]
    send_time: float


class Master(threading.Thread):
    def __init__(self, wk_num: int, raw_dnn: RawDNN, video_path: str, frame_size: Tuple[int, int],
                 job_type: Type[Job], check: bool, stb_fct: MStubFactory, config: Dict[str, Any]):
        super().__init__(daemon=True)
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__stb_fct = stb_fct
        self.__ifr_num = config['ifr_num']
        self.__itv_time = config['itv_time']
        self.__logger.info("Profiling data sizes...")
        self.__frame_size = frame_size
        self.__raw_dnn: Optional[RawDNN] = (raw_dnn if check else None)
        self.__pd_num = (config['pd_num'] if config['pd_num'] > 0 else float('inf'))
        self.__pd_dct = {}
        self.__pd_cv = threading.Condition()
        self.__begin_time = -1  # IFR0发出的时间
        self.__fin_ids = set()  # 目前所有已完成的IFR id，用于判断是否所有IFR已完成
        self.__vid_cap = cv2.VideoCapture(video_path)
        self.__init_scheduler(wk_num, raw_dnn, job_type, config)  # 初始化Scheduler会用到其他参数，所以最后执行
        self.__logger.info("Master init finished")

    def run(self) -> None:
        ifr_cnt = 0
        pre_ipt = torch.zeros(self.__frame_size)
        while self.__vid_cap.isOpened() and ifr_cnt < self.__ifr_num:
            gp_size = self.__scheduler.group_size()
            ipt_group = [self.get_ipt_from_video(self.__vid_cap, self.__frame_size)
                     for _ in range(min(gp_size, self.__ifr_num-ifr_cnt)) if self.__vid_cap.isOpened()]
            ifr_group = self.__scheduler.gen_ifr_group(ifr_cnt, pre_ipt, ipt_group)
            self.__send_group(ipt_group, ifr_group)
            pre_ipt = ipt_group[-1]
            ifr_cnt += len(ipt_group)
            time.sleep(self.__itv_time)

    def report_finish(self, ifr_id: int, tensor: Tensor = None) -> None:
        with self.__pd_cv:
            # 因为pd_ipt一定在里面，所以不会阻塞
            pd_ipt = self.__pd_dct.pop(ifr_id)
            self.__fin_ids.add(ifr_id)
            self.__pd_cv.notifyAll()
        self.__logger.info(f"IFR{ifr_id} finished, latency={time.time()-pd_ipt.send_time}s")
        if len(self.__fin_ids) == self.__ifr_num - 1:  # 所有IFR均完成
            # 注意：因为调度策略可能会变化，所以IFR0可能在w0完成，而IFR1可能在w1完成，从而导致IFR可能不是按序完成的
            # 但是，同一个IFR在worker间的执行顺序是固定的，所以相邻Worker的缓存应该是可以保证一致的
            self.__logger.info(f"All {self.__ifr_num} IFRs finished, "
                               f"avg cost={(time.time()-self.__begin_time)/self.__ifr_num}s")
        if self.__raw_dnn is not None:
            assert tensor is not None, "check is True but result is None!"
            self.__logger.info(f"checking IFR{ifr_id}")
            results = self.__raw_dnn.execute(pd_ipt.ipt)
            err = torch.max(torch.abs(tensor-results[-1]))
            if err < 1e-5:
                self.__logger.info(f"IFR{ifr_id} max_err={err}")
            else:
                self.__logger.warning(f"IFR{ifr_id} max_err={err} > 1e-5!")

    def __init_scheduler(self, wk_num: int, raw_dnn: RawDNN, job_type: Type[Job], config: Dict[str, Any]) -> None:
        # 加载DAG，获取predictor
        s_dag = cached_func(f"{raw_dnn.dnn_cfg.name}.{self.__frame_size[0]}x{self.__frame_size[1]}.sz",
                            SizedNode.raw2dag_sized, raw_dnn, self.__frame_size, logger=self.__logger)
        self.__logger.info(f"Getting predictors from trainer...")
        predictors = self.__stb_fct.trainer().get_predictors()
        # 获取worker计算能力及耗时
        wk_costs = [[] for _ in range(wk_num)]
        for wid in range(wk_num):
            self.__logger.info(f"Getting layer costs from worker{wid}...")
            wk_costs[wid] = self.__stb_fct.worker(wid).layer_cost()
        base_wk = 0  # 编号最小的作为计算能力的baseline
        wk_cap = []  # worker_id->相对计算能力
        for wk, costs in enumerate(wk_costs):
            assert costs[0] == 0, f"InputModule of Worker{wk} cost should be 0!"
            # Worker计算能力：基准worker的总耗时 / 当前worker的总耗时
            wk_cap.append(sum(wk_costs[base_wk]) / sum(costs))
        self.__logger.info(f"baseline=w{base_wk}, wk_cap={wk_cap}")
        ly_comp = wk_costs[base_wk]  # 各层计算能力，以base_wk为基准
        self.__logger.info(f"ly_comp={ly_comp}")
        wk_bwth = [bw * 1024 * 1024 for bw in config['bandwidth']]  # 单位MB转成B
        # 构造Scheduler
        schd_type = config['scheduler']
        self.__scheduler = schd_type(s_dag, predictors, wk_cap, wk_bwth, ly_comp, job_type, self.__ifr_num,
                                     defaultdict(dict, config)[schd_type.__name__])

    def __send_group(self, ipt_group: List[Tensor], ifr_group: List[IFR]):
        for i, ifr in enumerate(ifr_group):
            self.__logger.info(f"ready IFR{ifr.id}: "
                               + ', '.join(f'w{wj.worker_id}={wj.job.exec_ids}' for wj in ifr.wk_jobs))
            pd_data = (ipt_group[i] if self.__raw_dnn is not None else None)
            pd_ipt = PendingIpt(ifr.id, pd_data, -1)
            # 可能因为pending数量达到上限而阻塞
            with self.__pd_cv:
                while len(self.__pd_dct) >= self.__pd_num:
                    self.__pd_cv.wait()
                self.__pd_dct[pd_ipt.ifr_id] = pd_ipt
                self.__pd_cv.notifyAll()
            self.__logger.info(f"start process IFR{ifr.id}", extra={'trace': True})
            pd_ipt.send_time = time.time()
            self.__stb_fct.worker(ifr.wk_jobs[0].worker_id).new_ifr(ifr)
            if ifr.id == 0:
                self.__begin_time = pd_ipt.send_time

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
