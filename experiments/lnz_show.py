"""读取一个LFCNZ文件，显示各层输出差值的稀疏情况
横轴为层号，纵轴为当前层输出和前一帧差值的非零占比
"""
import pickle
from typing import List

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

from core.dnn_config import RawLayer
from core.raw_dnn import RawDNN
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50


plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
# sm = FontProperties(size=16)
lg = FontProperties(size=16)


def lfcnz2lfnz(lfcnz: List[List[List[float]]]) -> List[List[float]]:
    """对于每个层的输出数据，把各通道的非零占比合并成整体非零占比"""
    return [[sum(cnz)/len(cnz) for cnz in fcnz] for fcnz in lfcnz]


def frame_seq(r_layers: List[RawLayer], lfnz: List[List[float]], thres: float = .495):
    """点的颜色表示该帧在视频中所处位置，thres为稀疏阈值"""
    nlayer = len(lfnz)  # 层数
    nframe = len(lfnz[0])  # 帧数
    ax = plt.gca()
    plt.tick_params(labelsize=13)
    ax.set_ylim(-.05, 1.05)
    # 确保xtick都是int
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('CNN层号', fontproperties=lg)
    ax.set_ylabel('非零元素占比', fontproperties=lg)
    for l in range(len(r_layers)):
        plt.scatter([l] * nframe, lfnz[l], s=5,
                    c=[i / nframe for i in range(nframe)],
                    marker='.', cmap='viridis')
        if nlayer >= 60:  # 超过60层，画垂直辅助线
            # 垂直辅助线：从1到最大值
            plt.plot([l, l], [1, max(lfnz[l])], color=(0.8, 0.8, 0.8), linestyle=':')
    # plt.plot([ax.get_xbound()[0], ax.get_xbound()[1]], [thres, thres], linestyle='--')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('播放进度', fontproperties=lg)
    cbar.set_ticks([0, .2, .4, .6, .8, 1])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', 'end'])
    plt.tight_layout()


def heatmap(r_layers: List[RawLayer], lfnz: List[List[float]], thres: float = .495):
    """点的颜色表示这个稀疏率出现次数，thres为稀疏阈值"""
    nlayer = len(lfnz)  # 层数
    nframe = len(lfnz[0])  # 帧数
    x, y = [], []  # 每个点为一帧，x为该帧数据所在层，y为该帧数据的非零占比
    for l in range(len(r_layers)):
        x.extend([l] * nframe)
        y.extend(lfnz[l])
    sps_cnt = 0
    for yelm in y:
        if yelm < .5:
            sps_cnt += 1
    xedges = [-0.5] + [l+0.5 for l in range(nlayer)]
    yedges = [0.] + [1/nlayer*l for l in range(1, nlayer+1)]
    plt.tick_params(labelsize=13)
    plt.hist2d(x, y, bins=(xedges, yedges), cmap='Reds')
    # plt.plot([0, nlayer - 1], [thres, thres], linestyle='--')
    print(f'sparse={sps_cnt}/{nframe*nlayer}={round(sps_cnt/(nframe*nlayer)*100, 2)}%')
    plt.gca().set_xlabel('CNN层号', fontproperties=lg)
    plt.gca().set_ylabel('非零元素占比', fontproperties=lg)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('帧数', fontproperties=lg)


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
    with open(f"dataset/{CNN_NAME}.{VIDEO_NAME}.{RESOLUTION}.{NFRAME_TOTAL}.lfcnz", 'rb') as f:
        g_lfcnz = pickle.load(f)
    g_lfcnz = [fcnz[:NFRAME_SHOW] for fcnz in g_lfcnz]
    g_lfnz = lfcnz2lfnz(g_lfcnz)
    frame_seq(g_r_layers, g_lfnz)
    plt.figure()
    heatmap(g_r_layers, g_lfnz)
    plt.show()
