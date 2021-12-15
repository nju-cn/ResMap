import logging
import os
from threading import Condition, Thread
from typing import Tuple, List, Any, Dict

import cv2
import tqdm
from torch import Tensor

from core.util import cached_func
from master.master import Master
from core.predictor import Predictor
from core.raw_dnn import RawDNN


class Trainer(Thread):
    """运行在较高性能和较大内存的PC上，收集数据并训练稀疏率预测模型"""

    def __init__(self, raw_dnn: RawDNN, video_path: str, frame_size: Tuple[int, int], config: Dict[str, Any]):
        super().__init__()
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__raw_dnn = raw_dnn
        self.__video_path = video_path
        self.__frame_size = frame_size
        self.__frame_num = config['frame_num']
        self.__cv = Condition()
        self.__predictors = []

    def run(self) -> None:
        cnn_name = self.__raw_dnn.dnn_cfg.name
        vid_name = os.path.basename(self.__video_path).split('.')[0]  # 只保留文件名，去掉拓展名
        frm_size = f"{self.__frame_size[0]}x{self.__frame_size[1]}"
        trn_numb = str(self.__frame_num)
        lfcnz_path = cnn_name + '.' + vid_name + '.' + frm_size + '.' + trn_numb + '.lfcnz'
        self.__logger.info("collecting LFCNZ data...")
        lfcnz = cached_func(lfcnz_path, self.collect_lfcnz, self.__raw_dnn, self.__video_path,
                            self.__frame_num, self.__frame_size, logger=self.__logger)
        pred_path = cnn_name + '.' + vid_name + '.' + frm_size + '.' + trn_numb + '.pred'
        self.__logger.info("training predictors...")
        predictors = cached_func(pred_path, self.train_predictors, self.__raw_dnn, lfcnz, logger=self.__logger)
        self.__logger.info("train finished, predictors are ready")
        with self.__cv:
            self.__predictors = predictors
            self.__cv.notifyAll()

    def get_predictors(self) -> List[Predictor]:
        self.__logger.debug("trying to get predictors...")
        with self.__cv:
            while len(self.__predictors) == 0:
                self.__cv.wait()
            self.__logger.debug("got predictors")
            return self.__predictors

    @classmethod
    def collect_lfcnz(cls, raw_dnn: RawDNN, video_path: str,
                      frame_num: int, frame_size: Tuple[int, int]) -> List[List[List[float]]]:
        """收集LFCNZ数据：data[层l][帧f][通道c] = (帧f-上一帧)在层l输出数据中通道c的非零占比nz"""
        cap = cv2.VideoCapture(video_path)
        lst_results = []
        lfcnz = [[] for _ in raw_dnn.layers]
        for _ in tqdm.tqdm(range(frame_num)):
            cur_results = raw_dnn.execute(Master.get_ipt_from_video(cap, frame_size))
            lcnz = cls._dif_lcnz(lst_results, cur_results)
            for l in range(len(lcnz)):
                lfcnz[l].append(lcnz[l])
            lst_results = cur_results
        return lfcnz

    @classmethod
    def train_predictors(cls, raw_dnn: RawDNN, lfcnz: List[List[List[float]]]) -> List[Predictor]:
        predictors = []
        for l in tqdm.tqdm(range(len(raw_dnn.layers))):
            layer = raw_dnn.layers[l]
            predictor = raw_dnn.dnn_cfg.mdl2pred[layer.module.__class__](layer.module)
            afcnz = [lfcnz[al.id_] for al in layer.ac_layers]
            predictor.fit(afcnz, lfcnz[layer.id_])
            predictors.append(predictor)
        return predictors

    @classmethod
    def _layer_cnz(cls, tensor4d: Tensor) -> List[float]:
        return [float(chan.count_nonzero() / chan.nelement()) for chan in tensor4d[0]]

    @classmethod
    def _dif_lcnz(cls, last_results: List[Tensor], cur_results: List[Tensor]) -> List[List[float]]:
        if len(last_results) == 0:
            return [cls._layer_cnz(res) for res in cur_results]
        dif_results = []
        for i in range(len(last_results)):
            dif_results.append(cls._layer_cnz(cur_results[i] - last_results[i]))
        return dif_results
