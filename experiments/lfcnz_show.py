"""读取LFCNZ格式的数据，可视化
"""
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from tqdm import tqdm

from core.executor import Node
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from core.raw_dnn import RawDNN
from master.scheduler import Scheduler
from trainer.trainer import Trainer


def chan_err_mtrx():
    # 逐层展示各通道的预测误差
    NFRAME_TRAIN = 200  # 前多少帧的数据用于训练

    norm = matplotlib.colors.Normalize(vmin=0, vmax=.2)  # colorbar范围统一
    for lid in range(1, len(raw_dnn.layers)):  # 从InputModule后面开始
        pedor = raw_dnn.dnn_cfg.mdl2pred[raw_dnn.layers[lid].module.__class__](raw_dnn.layers[lid].module)
        pedor.fit([lfcnz[al.id_][:NFRAME_TRAIN] for al in raw_dnn.layers[lid].ac_layers], lfcnz[lid][:NFRAME_TRAIN])
        fcnz_prd = []
        for f in range(NFRAME_TRAIN, NFRAME_TOTAL):
            fcnz_prd.append(pedor.predict([lfcnz[al.id_][f] for al in raw_dnn.layers[lid].ac_layers]))
        fcnz_prd = np.array(fcnz_prd)
        fcnz_trh = np.array(lfcnz[lid][NFRAME_TRAIN:NFRAME_TOTAL])
        fcnz_err = np.abs(fcnz_prd - fcnz_trh)
        # 可视化
        fig = plt.figure()
        # 窗口最大化，便于观察
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # 绘制图像
        plt.imshow(fcnz_err, norm=norm)
        plt.colorbar()
        fig.suptitle(f"Layer{lid}: " + raw_dnn.layers[lid].module_type())
        plt.show()


def layer_regr_curve():
    # 逐帧展示各层的整体误差
    NFRAME_TRAIN = 200  # 前多少帧的数据用于训练
    NFRAME_PRED = 100  # 对后面的多少帧进行预测
    ENABLE_FRAME = False

    print("training...", file=sys.stderr)
    predictors = Trainer.train_predictors(raw_dnn, [fcnz[:NFRAME_TRAIN] for fcnz in lfcnz])

    print("predicting...", file=sys.stderr)
    dag = Node.raw2dag(raw_dnn.layers)
    fl_err = []
    flnz_trh, flnz_prd = [], []
    for fid in tqdm(range(NFRAME_TRAIN, NFRAME_TRAIN+NFRAME_PRED)):
        # 真实值
        lnz_trh = [sum(lfcnz[l][fid])/len(lfcnz[l][fid]) for l in range(len(lfcnz))]
        flnz_trh.append(lnz_trh)
        cnz = lfcnz[0][fid]
        lcnz_prd = Scheduler.predict_dag(cnz, dag, predictors)
        lnz_prd = [sum(lcnz_prd[l])/len(lcnz_prd[l]) for l in range(len(lcnz_prd))]
        flnz_prd.append(lnz_prd)
        if ENABLE_FRAME:
            plt.plot(lnz_trh, 'r')
            plt.plot(lnz_prd, 'b')
            plt.title(f'frame{fid}')
            plt.show()
        # 计算误差
        fl_err.append(np.abs(np.array(lnz_prd) - np.array(lnz_trh)))
    fl_err = np.array(fl_err)
    # 误差图
    plt.figure()
    plt.imshow(fl_err)
    plt.colorbar()
    plt.title('error')
    plt.show()
    # 真实值
    plt.figure()
    plt.imshow(flnz_trh)
    plt.colorbar()
    plt.title('truth')
    plt.show()
    # 预测值
    plt.figure()
    plt.imshow(flnz_prd)
    plt.colorbar()
    plt.title('predict')
    plt.show()


if __name__ == '__main__':
    CNN_NAME = 'gn'
    VIDEO_NAME = 'road'
    RESOLUTION = '480x720'  # 数据集的分辨率
    LEVEL = 'ly'
    NFRAME_TOTAL = 400  # 数据集中的帧数

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    with open(f'dataset/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz', 'rb') as lfile:
        lfcnz = pickle.load(lfile)
    if LEVEL == 'ch':
        chan_err_mtrx()
    elif LEVEL == 'ly':
        layer_regr_curve()
