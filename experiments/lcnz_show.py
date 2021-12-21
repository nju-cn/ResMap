"""读取一个LFCNZ文件，对于指定的帧，显示每层输出数据差值不同通道的非零占比
横轴为CNN各层，纵轴为各个通道的非零占比
"""
import pickle

from matplotlib import pyplot as plt

from core.raw_dnn import RawDNN
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50


if __name__ == '__main__':
    CNN_NAME = 'vg16'
    VIDEO_NAME = 'road'
    RESOLUTION = '480x720'  # 数据集的分辨率
    NFRAME_TOTAL = 400  # 数据集中的帧数

    TARGET_FRAME = 1  # 展示 TARGET_FRAME 和 TARGET_FRAME-1 的输出数据差值

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    r_layers = raw_dnn.layers
    with open(f"dataset/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz", 'rb') as f:
        lfcnz = pickle.load(f)
    lcnz = [fcnz[TARGET_FRAME] for fcnz in lfcnz]
    for l in range(len(r_layers)):
        nchan = len(lcnz[l])
        plt.scatter([l]*nchan, lcnz[l])
    plt.show()
