"""研究各CNN层输出的3维矩阵中，各通道的稀疏程度
数据格式LFCNZ：data[层l][帧f][通道c] = (帧f-上一帧)在层l输出数据中通道c的非零占比nz
"""
import pickle

from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from trainer.trainer import Trainer

from core.raw_dnn import RawDNN
from dnn_models.chain import prepare_alexnet, prepare_vgg16


if __name__ == '__main__':
    CNN_NAME = 'vg16'
    VIDEO_NAME = 'campus'
    RESOLUTION = '480x720'  # 数据集的分辨率
    NFRAME_TOTAL = 400  # 数据集中的帧数

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    frame_size = tuple(map(int, RESOLUTION.split('x')))[:2]
    # noinspection PyTypeChecker
    lfcnz = Trainer.collect_lfcnz(raw_dnn, f'../media/{VIDEO_NAME}.mp4', NFRAME_TOTAL, frame_size)
    with open(f'dataset/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz', 'wb') as file:
        pickle.dump(lfcnz, file)
    print(f'dataset/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz has been written')
