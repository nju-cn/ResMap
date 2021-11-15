from typing import Callable, List

import cv2
import torch

from dnn_config import DNNConfig
from dnn_models.chain import prepare_alexnet, prepare_vgg19
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from integral_executor import IntegralExecutor, IntegralJob
from raw_dnn import RawDNN
from unit_tests.common import get_ipt_from_video


def _run_jobs(raw_dnn: RawDNN, wk2jobs: List[IntegralJob]):
    cap = cv2.VideoCapture(f'../media/road.mp4')
    ipt = get_ipt_from_video(cap)
    # 注意：第0层必须是InputModule
    wk2jobs[0].id2opt[0] = ipt
    executor = IntegralExecutor(raw_dnn)
    ex_out = {}
    for wk, cur_job in enumerate(wk2jobs):
        id2opt = executor.exec(cur_job)
        if wk + 1 < len(wk2jobs):
            wk2jobs[wk + 1].id2opt = id2opt
        else:
            ex_out = id2opt
    # 直接执行RawLayer，以检查正确性
    results = raw_dnn.execute(ipt)
    assert len(ex_out) == 1, "There is more than 1 output node!"
    assert torch.allclose(results[-1], next(iter(ex_out.values())), atol=1e-6), \
        f"max_err={torch.max(torch.abs(results[-1]-next(iter(ex_out.values()))))}"


def test_alexnet():
    wk2jobs = [IntegralJob(list(range(1, 5)), [4], {}),
               IntegralJob(list(range(5, 10)), [9], {}),
               IntegralJob(list(range(10, 14)), [13], {})]
    _run_jobs(RawDNN(prepare_alexnet()), wk2jobs)


def test_vgg19():
    wk2jobs = [IntegralJob(list(range(1, 10)), [9], {}),
               IntegralJob(list(range(10, 20)), [19], {}),
               IntegralJob(list(range(20, 38)), [37], {})]
    _run_jobs(RawDNN(prepare_vgg19()), wk2jobs)


def test_googlenet():
    wk2jobs = [IntegralJob(list(range(1, 55)) + list(range(55, 57)) + list(range(59, 63)) + list(range(65, 68)),
                           [56, 62, 67, 55], {}),
               IntegralJob(list(range(57, 59)) + list(range(63, 65)) + list(range(68, 71)) + list(range(71, 118))
                           + list(range(118, 122)) + list(range(122, 124)) + list(range(128, 130)) + list(range(134, 135)),
                           [121, 123, 129, 134], {}),
               IntegralJob(list(range(124, 128)) + list(range(130, 134)) + list(range(135, 203)), [202], {})]
    _run_jobs(RawDNN(prepare_googlenet()), wk2jobs)


def test_resnet50():
    wk2jobs = [IntegralJob(list(range(1, 55)), [49, 54], {}),
               IntegralJob(list(range(55, 110)), [101, 109], {}),
               IntegralJob(list(range(110, 173)), [172], {})]
    _run_jobs(RawDNN(prepare_resnet50()), wk2jobs)
