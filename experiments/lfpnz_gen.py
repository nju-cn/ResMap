"""研究各CNN层输出的3维矩阵中，各平面的稀疏程度
数据格式LFPNZ：data[层l][帧f][平面p] = (帧f-上一帧)在层l输出数据中平面p的非零占比nz
"""
import pickle
from typing import List
import os

from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from torch import Tensor
import matplotlib.pyplot as plt

from raw_dnn import RawDNN
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from unit_tests.common import get_ipt_from_video


def layer_pnz(tensor4d: Tensor) -> List[float]:
    return [float(plane.count_nonzero()/plane.nelement()) for plane in tensor4d[0]]


def dif_lpnz(last_results: List[Tensor], cur_results: List[Tensor]) -> List[List[float]]:
    if len(last_results) == 0:
        return [layer_pnz(res) for res in cur_results]
    dif_results = []
    for i in range(len(last_results)):
        dif_results.append(layer_pnz(cur_results[i] - last_results[i]))
    return dif_results


if __name__ == '__main__':
    VIDEO_NAME = 'road'
    CNN_NAME = 'vg16'
    FRAME_NUM = 500  # 获取多少帧的数据
    PLOT = False  # 是否可视化
    WRITE = True  # 是否写入文件

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    cap = cv2.VideoCapture(f'../media/{VIDEO_NAME}.mp4')
    cnt = 0
    lst_res = []
    lfpnz = [[] for _ in raw_dnn.layers]
    while cap.isOpened() and cnt < FRAME_NUM:
        print(f"processing frame{cnt}")
        cur_res = raw_dnn.execute(get_ipt_from_video(cap))
        lpnz = dif_lpnz(lst_res, cur_res)
        if PLOT:
            for l in range(len(lpnz)):
                plt.scatter([l]*len(lpnz[l]), lpnz[l])
            plt.show()
        for l in range(len(lpnz)):
            lfpnz[l].append(lpnz[l])
        lst_res = cur_res
        cnt += 1
    print(f"nframe={len(lfpnz[1])}, pnz={lfpnz[1][1]}")
    if WRITE:
        with open(f'dataset/{VIDEO_NAME}.{CNN_NAME}.lfpnz', 'wb') as file:
            pickle.dump(lfpnz, file)
