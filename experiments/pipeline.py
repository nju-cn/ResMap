from typing import List, Tuple

import cv2
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from core.raw_dnn import RawDNN
from core.util import cached_func
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from master.master import Master
from master.scheduler import Scheduler, SizedNode
from trainer.trainer import Trainer


def visualize_frames(wk_tran: List[float], wk_cmpt: List[float], nframe: int):
    """wk_tran[w], wk_cmpt[w]表示Worker w的传输耗时和计算耗时，先传输后计算"""
    # 计算耗时情况
    dp = Scheduler.simulate_pipeline(wk_tran, wk_cmpt, nframe)
    print(dp)
    # 可视化
    fig = plt.figure()
    ax = fig.subplots()
    ax.invert_yaxis()
    ticklabels = ['m->w0', 'w0']
    for w in range(1, len(wk_cmpt)):
        ticklabels.extend([f'{w-1}->{w}', f'w{w}'])
    plt.yticks(list(range(2*len(wk_cmpt))), ticklabels)
    colors = list(mcolors.XKCD_COLORS.values())
    for f in range(nframe):
        for w in range(len(wk_cmpt)):
            plt.barh(2*w, wk_tran[w], left=dp[f][2*w]-wk_tran[w], color=colors[f])
            plt.barh(2*w+1, wk_cmpt[w], left=dp[f][2*w+1]-wk_cmpt[w], color=colors[f])
    plt.show()


if __name__ == '__main__':
    CNN_NAME = 'ax'
    VIDEO_NAME = 'road'
    RESOLUTION = '480x720'
    NFRAME_TOTAL = 400
    NFRAME_SHOW = 5

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    # AlexNet在PC上的耗时
    ax_pc = [0, 0.017154693603515625, 0.002193784713745117, 0.010571908950805665,
             0.01097102165222168, 0.000997018814086914, 0.006582069396972656,
             0.006582736968994141, 0.0005986213684082032, 0.008776283264160157,
             0.000399017333984375, 0.005983781814575195, 0.0007979393005371094, 0.0011966705322265625]
    wk_lynum = Scheduler.split_chain(ax_pc, [1, 1, 1])
    lb_wk_layers = Scheduler.wk_lynum2layers_chain(1, wk_lynum)

    cap = cv2.VideoCapture(f'../media/{VIDEO_NAME}.mp4')
    frame_size = tuple(map(int, RESOLUTION.split('x')))
    ipt = Master.get_ipt_from_video(cap, frame_size)
    cnz = [float(chan.count_nonzero() / chan.nelement()) for chan in ipt[0]]
    dag = cached_func(f"{CNN_NAME}.{RESOLUTION}.sz", SizedNode.raw2dag_sized, raw_dnn, frame_size)

    lfcnz_path = '' + CNN_NAME + '.' + VIDEO_NAME + '.' + RESOLUTION + '.' + str(NFRAME_TOTAL) + '.lfcnz'
    print("collecting LFCNZ data...")
    lfcnz = cached_func(lfcnz_path, Trainer.collect_lfcnz, raw_dnn, f'../media/{VIDEO_NAME}.mp4',
                        NFRAME_TOTAL, frame_size)
    pred_path = CNN_NAME + '.' + VIDEO_NAME + '.' + RESOLUTION + '.' + str(NFRAME_TOTAL) + '.pred'
    print("training predictors...")
    predictors = cached_func(pred_path, Trainer.train_predictors, raw_dnn, lfcnz)
    lcnz = Scheduler.predict_dag(cnz, dag, predictors)
    lsz = Scheduler.lcnz2lsz(lcnz, dag)
    lbsz = [sz * 4 for sz in lsz]

    visualize_frames([1, 2, 3], [1.5, .5, 1], NFRAME_SHOW)
