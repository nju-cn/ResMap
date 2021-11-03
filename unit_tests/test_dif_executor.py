from typing import List

import cv2
import torch
from torch import Tensor

from dif_executor import DifExecutor, DifJob
from dnn_models.resnet import prepare_resnet50
from unit_tests.common import get_ipt_from_video


def _co_execute(frame_id: int, dif_extors: List[DifExecutor],
                wk2job: List[DifJob], benchmarks: List[Tensor]):
    for wk, d_job in enumerate(wk2job):
        id2dif = dif_extors[wk].exec(d_job)
        id2opt = dif_extors[wk].last_out()
        for oid in d_job.out_ids:
            assert torch.allclose(id2opt[oid], benchmarks[oid], atol=1e-5), \
                f"Frame{frame_id} Worker{wk} layer{oid} max_err={torch.max(torch.abs(id2opt[oid]-benchmarks[oid]))}"
        if wk + 1 < len(wk2job):
            wk2job[wk + 1].id2dif = id2dif


def test_fixed_jobs():
    """当DifJob固定时，测试多个DifExecutor是否可以正常协同，观察误差"""
    dif_extors = [DifExecutor(prepare_resnet50) for _ in range(3)]  # 分别对应3个Worker
    cap = cv2.VideoCapture(f'../media/road.mp4')
    cur = get_ipt_from_video(cap)
    dif = cur
    for fid in range(5):
        # 直接执行RawLayer，以检查正确性
        results = dif_extors[0].check_exec(cur)
        wk2job = [DifJob(list(range(1, 55)), [49, 54], {0: dif}),
                  DifJob(list(range(55, 110)), [101, 109], {}),
                  DifJob(list(range(110, 173)), [172], {})]
        _co_execute(fid, dif_extors, wk2job, results)
        nxt = get_ipt_from_video(cap)
        dif = nxt - cur
        cur = nxt


def test_var_jobs():
    """当DifJob会变时，测试多个DifExecutor是否可以正常协同，观察误差"""
    dif_extors = [DifExecutor(prepare_resnet50) for _ in range(3)]  # 分别对应3个Worker
    cap = cv2.VideoCapture(f'../media/road.mp4')
    cur = get_ipt_from_video(cap)
    dif = cur
    for fid in range(4):
        # 直接执行RawLayer，以检查正确性
        results = dif_extors[0].check_exec(cur)
        plans = [
                 # 初始计划
                 [DifJob(list(range(1, 55)), [49, 54], {0: dif}),
                  DifJob(list(range(55, 110)), [101, 109], {}),
                  DifJob(list(range(110, 173)), [172], {})],
                 # Worker0-1边界变化
                 [DifJob(list(range(1, 44)) + [46], [43, 46], {0: dif}),
                  DifJob([44, 45] + list(range(47, 110)), [101, 109], {}),
                  DifJob(list(range(110, 173)), [172], {})],
                 # Worker1-2边界变化
                 [DifJob(list(range(1, 44)) + [46], [43, 46], {0: dif}),
                  DifJob([44, 45] + list(range(47, 106)), [101, 105], {}),
                  DifJob(list(range(106, 173)), [172], {})],
                 # Worker0-1,1-2边界都变化了
                 [DifJob(list(range(1, 37)), [36], {0: dif}),
                  DifJob(list(range(37, 111)), [110], {}),
                  DifJob(list(range(111, 173)), [172], {})]
                 ]
        wk2job = plans[fid]
        _co_execute(fid, dif_extors, wk2job, results)
        nxt = get_ipt_from_video(cap)
        dif = nxt - cur
        cur = nxt