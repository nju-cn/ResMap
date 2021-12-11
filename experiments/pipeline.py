import cv2

from core.raw_dnn import RawDNN
from core.util import cached_func
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from master.master import Master
from master.scheduler import Scheduler, SizedNode
from trainer.trainer import Trainer


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
    # AlexNet在pi4G上的耗时
    ax_pc = [0.0, 0.2958364963531494, 0.008954572677612304, 0.05184073448181152, 0.3531335830688477,
             0.0068168163299560545, 0.03638005256652832, 0.15417227745056153, 0.003465700149536133,
             0.2227616786956787, 0.0024199485778808594, 0.1506051540374756, 0.0024116992950439452,
             0.010980749130249023]
    wk_cap = [1.0, 16.12921131905995, 12.796705048150164]  # pi4G, PC, aliyun
    wk_bwth = [5.34, 7.61, 1.66]  # 单位MB
    wk_bwth = [bw*1024*1024 for bw in wk_bwth]

    wk_lynum = Scheduler.split_chain(ax_pc[1:], wk_cap)
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
    Scheduler.optimize_chain(lb_wk_layers, wk_cap, wk_bwth, lbsz, ax_pc, 1, True)
