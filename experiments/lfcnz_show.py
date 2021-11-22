"""读取LFCNZ格式的数据，可视化
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from core.raw_dnn import RawDNN


if __name__ == '__main__':
    VIDEO_NAME = 'road'
    CNN_NAME = 'ax'
    NFRAME_TRAIN = 400  # 前多少帧的数据用于训练
    PLOT = True  # 是否对每一层的预测误差都绘制图像

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    norm = matplotlib.colors.Normalize(vmin=0, vmax=.2)  # colorbar范围统一
    with open(f'dataset/{VIDEO_NAME}.{CNN_NAME}.lfcnz', 'rb') as f:
        lfcnz = pickle.load(f)
    nframe = len(lfcnz[0])
    for lid in range(1, len(raw_dnn.layers)):  # 从InputModule后面开始
        pedor = raw_dnn.dnn_cfg.mdl2pred[raw_dnn.layers[lid].module.__class__](raw_dnn.layers[lid].module)
        pedor.fit([lfcnz[al.id_][:NFRAME_TRAIN] for al in raw_dnn.layers[lid].ac_layers], lfcnz[lid][:NFRAME_TRAIN])
        fcnz_prd = []
        for f in range(NFRAME_TRAIN, nframe):
            fcnz_prd.append(pedor.predict([lfcnz[al.id_][f] for al in raw_dnn.layers[lid].ac_layers]))
        fcnz_prd = np.array(fcnz_prd)
        fcnz_trh = np.array(lfcnz[lid][NFRAME_TRAIN:nframe])
        fcnz_err = np.abs(fcnz_prd - fcnz_trh)
        if PLOT:
            fig = plt.figure()
            # 窗口最大化，便于观察
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            # 绘制图像
            plt.imshow(fcnz_err, norm=norm)
            plt.colorbar()
            fig.suptitle(f"Layer{lid}: " + raw_dnn.layers[lid].module_type())
            plt.show()