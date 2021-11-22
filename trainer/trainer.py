import os
import pickle
from threading import Condition, Thread
from typing import Tuple, List, Callable, Any, Dict

import cv2
import tqdm
from torch import Tensor

from core.dnn_config import DNNConfig
from master.master import Master
from core.predictor import Predictor
from core.raw_dnn import RawDNN


class Trainer(Thread):
    """运行在较高性能和较大内存的PC上，收集数据并训练稀疏率预测模型"""
    CNN_ABR = {}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.cv = Condition()
        self.predictors = []

    def run(self) -> None:
        raw_dnn = RawDNN(self.config['dnn_loader']())
        cnn_name = self.dnn_abbr(self.config['dnn_loader'])  # ax, gn等
        vid_name = os.path.basename(self.config['video_path']).split('.')[0]  # 只保留文件名，去掉拓展名
        frm_size = f"{self.config['frame_size'][0]}x{self.config['frame_size'][1]}"
        trn_numb = str(self.config['trainer']['frame_num'])
        lfcnz_path = cnn_name + '.' + vid_name + '.' + frm_size + '.' + trn_numb + '.lfcnz'
        print("collecting LFCNZ data...")
        lfcnz = self.cached_func(lfcnz_path, self.collect_lfcnz, raw_dnn, self.config['video_path'],
                                 self.config['trainer']['frame_num'], self.config['frame_size'])
        pred_path = cnn_name + '.' + vid_name + '.' + frm_size + '.' + trn_numb + '.pred'
        print("training predictors...")
        predictors = self.cached_func(pred_path, self.train_predictors, raw_dnn, lfcnz)
        print("train finished, predictors are ready")
        with self.cv:
            self.predictors = predictors
            self.cv.notifyAll()

    def get_predictors(self) -> List[Predictor]:
        print("trying to get predictors...")
        with self.cv:
            while len(self.predictors) == 0:
                self.cv.wait()
            print("got predictors")
            return self.predictors

    _DNN_ABR = {'alexnet': 'ax', 'vgg16': 'vg16', 'googlenet': 'gn', 'resnet50': 'rs50'}

    @classmethod
    def dnn_abbr(cls, dnn_loader: Callable[[], DNNConfig]) -> str:
        return cls._DNN_ABR[dnn_loader.__name__.replace('prepare_', '')]

    @classmethod
    def cached_func(cls, cache_path: str, func: Callable, *args) -> Any:
        if os.path.isfile(cache_path):
            print(f"{cache_path} exists, loading...")
            with open(cache_path, 'rb') as cfile:
                return pickle.load(cfile)
        else:
            print(f"{cache_path} not exists, generating...")
            data = func(*args)
            print(f"{cache_path} generated, writing...")
            with open(cache_path, 'wb') as cfile:
                pickle.dump(data, cfile)
            return data

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
