"""读取一个LFCNZ文件，显示对于每一帧，各层输出数据的 实际非零占比 和 预测非零占比
每张子图是一个视频帧，横轴为各CNN层，纵轴为实际非零占比 和 预测非零占比
"""
import pickle
import sys
from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit

from core.executor import Node
from core.predictor import Predictor
from core.raw_dnn import RawDNN
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from schedulers.nsc_scheduler import NSCScheduler
from trainer.trainer import Trainer


class LgiPredictor(Predictor):
    def __init__(self, module: torch.nn.Module):
        super().__init__(module)
        self.popt = None
        self.nc = 0  # 输出channel数量

    def fit(self, afcnz: List[List[List[float]]], fcnz: List[List[float]]) -> 'Predictor':
        assert len(afcnz) == 1
        i_fnz = [sum(cnz)/len(cnz) for cnz in afcnz[0]]
        o_fnz = [sum(cnz)/len(cnz) for cnz in fcnz]
        xarr, yarr = list(zip(*sorted(zip(i_fnz, o_fnz))))
        xarr, yarr = np.array(xarr), np.array(yarr)
        popt, pcov = curve_fit(self.logistic, xarr, yarr, maxfev=50000)
        self.popt = popt
        self.nc = len(fcnz[0])
        return self

    def predict(self, acnz: List[List[float]]) -> List[float]:
        # 这里不需要cnz，所以把所有输出都填为同一个值
        assert len(acnz) == 1
        i_nz = sum(acnz[0]) / len(acnz[0])
        o_nz = self.logistic(i_nz, *self.popt)
        return [o_nz for _ in range(self.nc)]

    @staticmethod
    def logistic(x, k, p, r):
        return (k * p * np.exp(r * x)) / (k + p * (np.exp(r * x) - 1))  # logistic函数


if __name__ == '__main__':
    CNN_NAME = 'gn'
    VIDEO_NAME = 'road'
    RESOLUTION = '480x720'  # 数据集的分辨率
    NFRAME_TOTAL = 400  # 数据集中的帧数
    NFRAME_TRAIN = 300  # 用于训练的帧数，后面的都用于测试

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    with open(f"dataset/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz", 'rb') as f:
        g_lfcnz = pickle.load(f)

    # 用前NFRAME_TRAIN帧训练
    print("training predictors...", file=sys.stderr)
    mlp_preds = Trainer.train_predictors(raw_dnn, [fcnz[:NFRAME_TRAIN] for fcnz in g_lfcnz])
    lgi_preds = deepcopy(mlp_preds)
    for layer in raw_dnn.layers:
        # 把Conv2d的改成LgiPredictor，其余不变
        if not isinstance(layer.module, torch.nn.Conv2d):
            continue
        lgi_preds[layer.id_] = LgiPredictor(layer.module)
        afcnz = [g_lfcnz[al.id_] for al in layer.ac_layers]
        lgi_preds[layer.id_].fit(afcnz, g_lfcnz[layer.id_][:NFRAME_TRAIN])
    # 用NFRAME_TRAIN后的所有帧检测
    dag = Node.raw2dag(raw_dnn.layers)
    for f in range(NFRAME_TRAIN, NFRAME_TOTAL):
        ipt_cnz = g_lfcnz[0][f]
        mlp_lcnz = NSCScheduler.predict_dag(g_lfcnz[0][f], dag, mlp_preds)
        mlp_lnz = [sum(cnz)/len(cnz) for cnz in mlp_lcnz]
        ipt_nz = sum(ipt_cnz)/len(ipt_cnz)
        lgi_lcnz = NSCScheduler.predict_dag([ipt_nz], dag, lgi_preds)
        lgi_lnz = [cnz[0] for cnz in lgi_lcnz]
        gt_lnz = [sum(fcnz[f])/len(fcnz[f]) for fcnz in g_lfcnz]
        plt.plot(list(range(len(dag))), gt_lnz, '*')
        plt.plot(list(range(len(dag))), mlp_lnz, 'r-')
        plt.plot(list(range(len(dag))), lgi_lnz, 'b-')
        plt.title(f"mlp_err={round(float(np.sum(np.abs(np.array(mlp_lnz)-np.array(gt_lnz)))), 2)}, "
                  f"lgi_err={round(float(np.sum(np.abs(np.array(lgi_lnz)-np.array(gt_lnz)))), 2)}")
        plt.show()
