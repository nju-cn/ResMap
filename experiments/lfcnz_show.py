"""读取LFCNZ格式的数据，可视化
"""
import pickle
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet, InceptionCat
from dnn_models.resnet import prepare_resnet50, BottleneckAdd
from raw_dnn import RawDNN


if __name__ == '__main__':
    VIDEO_NAME = 'road'
    CNN_NAME = 'ax'
    PLOT = True  # 是否对每一层的预测误差都绘制图像

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    norm = matplotlib.colors.Normalize(vmin=0, vmax=.2)  # colorbar范围统一
    with open(f'dataset/{VIDEO_NAME}.{CNN_NAME}.lfcnz', 'rb') as f:
        lfcnz = pickle.load(f)
    for lid in range(1, len(raw_dnn.layers)):  # 从InputModule后面开始
        if PLOT:
            fig = plt.figure()
            # 窗口最大化，便于观察
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        if len(raw_dnn.layers[lid].ac_layers) == 1:
            # X为多份输入样本，每份样本为 输入数据所有通道的非零占比（除去第0帧，因为它不是差值）
            # y为多份输出结果
            X = np.array(lfcnz[raw_dnn.layers[lid].ac_layers[0].id_][1:])
            y = np.array(lfcnz[lid][1:])
            # 一个感知机拟合所有通道
            begin = time.time()
            mlp = MLPRegressor((1,), activation='logistic', solver='lbfgs', max_iter=500).fit(X[:400], y[:400])
            mlp_err = np.abs(mlp.predict(X[400:]) - y[400:])
            print(f"Layer{lid} {raw_dnn.layers[lid].module_type()}: "
                  f"MLP cost {round(time.time() - begin, 4)}s", end='')
            if PLOT:
                ax = plt.subplot(1, 3, 1)
                im = ax.imshow(mlp_err, norm=norm)
                plt.colorbar(im)
                ax.set_title(f'MLP, mean={round(np.mean(mlp_err), 4)}')
            # 一个感知机拟合一个通道
            err = []
            for c in range(y.shape[1]):
                mlp = MLPRegressor((1,), activation='logistic', solver='lbfgs', max_iter=500).fit(X[:400], y[:400, c])
                err.append(np.abs(mlp.predict(X[400:]) - y[400:, c]))
            err = np.array(err).T
            if PLOT:
                ax = plt.subplot(1, 3, 2)
                im = ax.imshow(err, norm=norm)
                plt.colorbar(im)
                ax.set_title(f'MLPs, mean={round(np.mean(err), 4)}')
            # ReLU, MaxPool2d, BatchNorm2d：输出通道数=输入通道数，可以直接用线性函数拟合
            if not isinstance(raw_dnn.layers[lid].module, torch.nn.Conv2d):
                # 一次函数拟合
                begin = time.time()
                lnr_err = []
                for c in range(X.shape[1]):
                    lnr = LinearRegression().fit(X[:400, c].reshape(-1, 1), y[:400, c])
                    lnr_err.append(np.abs(lnr.predict(X[400:, c].reshape(-1, 1)) - y[400:, c]))
                print(f", LNR cost {round(time.time() - begin, 4)}s")
                if PLOT:
                    ax = plt.subplot(1, 3, 3)
                    lnr_err = np.array(lnr_err).T
                    im = ax.imshow(lnr_err, norm=norm)
                    plt.colorbar(im)
                    ax.set_title(f'LNR, mean={round(np.mean(lnr_err), 4)}')
            else:
                print()
        else:  # MergeModule有多个输入
            if isinstance(raw_dnn.layers[lid].module, InceptionCat):
                # InceptionCat输出的每一层都直接来自前面一层
                # y_pred根据先验知识直接预测，y_pred[帧号f][通道号p] = 非零占比nz
                y_pred = []
                for fid in range(1, len(lfcnz[lid])):
                    y_pred.append([])
                    for al in raw_dnn.layers[lid].ac_layers:
                        y_pred[-1].extend(lfcnz[al.id_][fid])
                y_pred = np.array(y_pred)
                y = np.array([lfcnz[lid][f] for f in range(1, len(lfcnz[lid]))])
                err = np.abs(y_pred - y)
                if PLOT:
                    ax = plt.subplot(1, 2, 1)
                    im = ax.imshow(err, norm=norm)
                    plt.colorbar(im)
                    ax.set_title(f'DRT, mean={round(np.mean(err), 4)}')
            elif isinstance(raw_dnn.layers[lid].module, BottleneckAdd):
                # 每个通道都有一个回归器
                begin = time.time()
                lnr_err = []
                for c in range(len(lfcnz[lid][0])):  # 遍历各通道
                    X = [[lfcnz[al.id_][f][c] for al in raw_dnn.layers[lid].ac_layers] for f in range(1, len(lfcnz[lid]))]
                    X = np.array(X)
                    y = np.array([lfcnz[lid][f][c] for f in range(1, len(lfcnz[lid]))])
                    lnr = LinearRegression().fit(X[:400], y[:400])
                    lnr_err.append(np.abs(lnr.predict(X[400:]) - y[400:]))
                print(f", LNR cost {round(time.time() - begin, 4)}s")
                if PLOT:
                    ax = plt.subplot(1, 2, 1)
                    lnr_err = np.array(lnr_err).T
                    im = ax.imshow(lnr_err, norm=norm)
                    plt.colorbar(im)
                    ax.set_title(f'LNR, mean={round(np.mean(lnr_err), 4)}')
            else:
                raise Exception(f"Unrecognized Module: {raw_dnn.layers[lid].module_type()}")
        if PLOT:
            fig.suptitle(f"Layer{lid}: " + raw_dnn.layers[lid].module_type())
            plt.show()
