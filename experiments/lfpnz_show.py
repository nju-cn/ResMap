"""读取LFPNZ格式的数据，可视化
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
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from raw_dnn import RawDNN


if __name__ == '__main__':
    VIDEO_NAME = 'road'
    CNN_NAME = 'vg16'
    PLOT = False  # 是否对每一层的预测误差都绘制图像

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    norm = matplotlib.colors.Normalize(vmin=0, vmax=.3)  # colorbar范围统一
    with open(f'dataset/{VIDEO_NAME}.{CNN_NAME}.lfpnz', 'rb') as f:
        lfpnz = pickle.load(f)
    for lid in range(1, len(raw_dnn.layers)):
        X = np.array(lfpnz[lid - 1][1:])  # 多份样本，每份样本为 输入数据所有平面的非零占比（除去第0帧，因为它不是差值）
        y = np.array(lfpnz[lid][1:])  # 输出数据
        # 感知机拟合
        begin = time.time()
        mlp = MLPRegressor((1,), activation='logistic', solver='lbfgs', max_iter=500).fit(X[:400], y[:400])
        mlp_err = np.abs(mlp.predict(X[400:]) - y[400:])
        print(f"Layer{lid} {raw_dnn.layers[lid].module_type()}: "
              f"MLP cost {round(time.time()-begin, 4)}s", end='')
        if PLOT:
            fig = plt.figure()
            ax = plt.subplot(1, 2, 1)
            im = ax.imshow(mlp_err, norm=norm)
            plt.colorbar(im)
            ax.set_title(f'MLP, mean={round(np.mean(mlp_err), 4)}')
        if not isinstance(raw_dnn.layers[lid].module, torch.nn.Conv2d):
            # 一次函数拟合
            begin = time.time()
            lnr_err = []
            for p in range(X.shape[1]):
                lnr = LinearRegression().fit(X[:400, p].reshape(-1, 1), y[:400, p])
                lnr_err.append(np.abs(lnr.predict(X[400:, p].reshape(-1, 1)) - y[400:, p]))
            print(f", LNR cost {round(time.time()-begin, 4)}s")
            if PLOT:
                ax = plt.subplot(1, 2, 2)
                lnr_err = np.array(lnr_err).T
                im = ax.imshow(np.array(lnr_err), norm=norm)
                plt.colorbar(im)
                ax.set_title(f'LNR, mean={round(np.mean(lnr_err), 4)}')
        else:
            print()
        if PLOT:
            fig.suptitle(f"Layer{lid}: " + raw_dnn.layers[lid].module_type())
            plt.show()
