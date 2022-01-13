"""读取一个LFCNZ文件，显示各层输出数据的数据量曲线：原始数据(不压缩/压缩)、差值压缩数据(实际值/预测值)
一个子图展示一个帧，横轴为层号，纵轴为数据量（单位MB）
"""
import pickle
from typing import List

from matplotlib import pyplot as plt
import numpy as np

from core.predictor import Predictor
from core.raw_dnn import RawDNN
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from master.scheduler import SizedNode, Scheduler
from schedulers.my_scheduler import MyScheduler

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号


def predict_show(s_dag: List[SizedNode], predictors: List[Predictor],
                 o_lcnz: List[List[float]], lfcnz: List[List[List[float]]]):
    SUB_NROW, SUB_NCOL = 3, 1
    nframe = len(lfcnz[0])
    nlayer = len(lfcnz)
    artery = MyScheduler.get_artery(s_dag)
    candidates = [at.id for at in artery if len(at.ancients) == 1]  # MyScheduler中的候选主干节点
    show_cads = False
    if nlayer != len(artery):  # 主干节点数!=总层数，说明有分叉
        show_cads = True
        print(f"DAG的主干节点: {candidates}")
    org_ucps = np.array([snd.out_size[0] * snd.out_size[1] * snd.out_size[2] for snd in s_dag])*4/1024/1024  # MB
    org_cps = np.array(Scheduler.lcnz2lsz(o_lcnz, s_dag))*4/1024/1024  # MB
    for f in range(nframe):
        if f//(SUB_NROW*SUB_NCOL) > 0 and f%(SUB_NROW*SUB_NCOL) == 0:
            plt.show()

        print(f"Frame{f}: 输入数据cnz={lfcnz[0][f]}")
        lcnz_pred = Scheduler.predict_dag(lfcnz[0][f], s_dag, predictors)
        dif_pred = np.array(Scheduler.lcnz2lsz(lcnz_pred, s_dag))*4/1024/1024  # 预测值, MB
        lcnz_gt = [lfcnz[l][f] for l in range(nlayer)]
        dif_gt = np.array(Scheduler.lcnz2lsz(lcnz_gt, s_dag))*4/1024/1024  # 实际值, MB

        ax = plt.subplot(SUB_NROW, SUB_NCOL, f%(SUB_NROW*SUB_NCOL)+1)
        ax.set_title(f'frame{f}')
        plt.plot(org_ucps, label='原始数据不压缩')
        plt.plot(org_cps, label='原始数据压缩')
        plt.plot(dif_pred, label='差值数据预测值')
        plt.plot(dif_gt, label='差值数据实际值')
        if show_cads:
            # 用垂直虚线标出候选主干节点
            for cad in candidates:
                plt.plot([cad, cad], [0, org_ucps.max()], 'b--', lw=1)
        plt.legend()


if __name__ == '__main__':
    CNN_NAME = 'ax'
    VIDEO_NAME = 'road'
    RESOLUTION = '480x720'  # 数据集的分辨率
    NFRAME_TOTAL = 400  # 数据集中的帧数
    NFRAME_SHOW = 400  # 展示数据集中的多少帧

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    g_r_layers = raw_dnn.layers
    with open(f"../.cache/{CNN_NAME}.{RESOLUTION}.sz", 'rb') as file:
        g_s_dag = pickle.load(file)
    with open(f"../.cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.pred", 'rb') as file:
        g_predictors = pickle.load(file)
    with open(f"../.cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.o_lcnz", 'rb') as file:
        g_o_lcnz = pickle.load(file)
    with open(f"../.cache/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz", 'rb') as file:
        g_lfcnz = pickle.load(file)
    g_lfcnz = [fcnz[:NFRAME_SHOW] for fcnz in g_lfcnz]
    predict_show(g_s_dag, g_predictors, g_o_lcnz, g_lfcnz)
