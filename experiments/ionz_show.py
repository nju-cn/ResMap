"""读取一个LFCNZ文件，显示每层的输入差值非零占比和输出差值非零占比的关系
每层是一个单独的图，横轴为输入差值非零占比，纵轴为输出差值非零占比
"""
import pickle
import sys
from typing import List, Type

import torch.nn
import tqdm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from core.dnn_config import RawLayer
from core.predictor import Predictor
from core.raw_dnn import RawDNN
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from trainer.trainer import Trainer


def lfcnz2lfnz(lfcnz: List[List[List[float]]]) -> List[List[float]]:
    """对于每个层的输出数据，把各通道的非零占比合并成整体非零占比"""
    return [[sum(cnz)/len(cnz) for cnz in fcnz] for fcnz in lfcnz]


def draw_predict(predictor: Predictor, i_fcnz: List[List[float]], ax: Axes):
    """对于特定的层，使用predictor预测各帧输出数据的总体非零占比，并绘制到ax上
    注意：这里只处理只有一个前驱的层，不处理多前驱的层
    """
    p_fnz = [0. for _ in i_fcnz]
    for f, cnz in enumerate(i_fcnz):
        p_cnz = predictor.predict([i_fcnz[f]])
        p_fnz[f] = sum(p_cnz) / len(p_cnz)
    i_fnz = [sum(cnz)/len(cnz) for cnz in i_fcnz]
    xarr, yarr = list(zip(*sorted(zip(i_fnz, p_fnz))))
    ax.plot(xarr, yarr, 'r-')


def target_layers_in_out(cnn_name: str, target_type: Type[torch.nn.Module], uni_scale: bool, show_seq: bool,
                  r_layers: List[RawLayer], lfcnz: List[List[List[float]]], predictors: List[Predictor] = None):
    """对于特定类型的所有层，显示输入和输出的关联。一个窗口展示3*5=15个层的数据
    target_type为要查看的layer类型
    uni_scale为是否把刻度统一到[0, 1]区间
    show_seq为是否用点的颜色表示帧的顺序
    """
    lfnz = lfcnz2lfnz(lfcnz)
    nframe = len(lfnz[0])
    cnt = 1
    print("plotting...", file=sys.stderr)
    for l in tqdm.tqdm(range(len(r_layers))):
        layer = r_layers[l].module
        if not isinstance(layer, target_type):
            continue
        xlabel = f"{cnn_name}-{l}"
        if target_type == torch.nn.Conv2d:
            xlabel += f": in={layer.in_channels}, out={layer.out_channels}"
        ax = plt.subplot(3, 5, cnt, xlabel=xlabel)
        if uni_scale:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        i_dif = lfnz[r_layers[l].ac_layers[0].id_]
        o_dif = lfnz[l]
        if show_seq:
            plt.scatter(i_dif, o_dif, s=2,
                        c=[i / nframe for i in range(nframe)],
                        marker='.', cmap='viridis')
        else:
            plt.scatter(i_dif, o_dif, s=2)
        if predictors is not None:
            draw_predict(predictors[l], lfcnz[r_layers[l].ac_layers[0].id_], ax)
        cnt += 1
        if cnt > 15:
            cnt = 1
            plt.figure()
    plt.show()


if __name__ == '__main__':
    CNN_NAME = 'ax'
    VIDEO_NAME = 'road'
    RESOLUTION = '480x720'  # 数据集的分辨率
    NFRAME_TOTAL = 400  # 数据集中的帧数

    NFRAME_SHOW = NFRAME_TOTAL  # 展示数据集中的多少帧
    TARGET_TYPE = 'cv'
    UNI_SCALE = True  # 是否统一刻度到[0, 1]区间
    SEQ_FRAME = False  # 是否用点的颜色表示帧的顺序
    PREDICT = True  # 是否显示对NFRAME_SHOW的预测图线（训练集也是NFRAME_SHOW）

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    target_types = {'cv': torch.nn.Conv2d, 'rl': torch.nn.ReLU, 'mp': torch.nn.MaxPool2d}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    g_r_layers = raw_dnn.layers
    with open(f"dataset/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz", 'rb') as f:
        g_lfcnz = pickle.load(f)
    g_lfcnz = [fcnz[:NFRAME_SHOW] for fcnz in g_lfcnz]

    if PREDICT:
        print("training...", file=sys.stderr)
        g_predictors = Trainer.train_predictors(raw_dnn, [fcnz[:NFRAME_SHOW] for fcnz in g_lfcnz])
    else:
        g_predictors = None
    target_layers_in_out(CNN_NAME, target_types[TARGET_TYPE], UNI_SCALE, SEQ_FRAME,
                         g_r_layers, g_lfcnz, g_predictors)
