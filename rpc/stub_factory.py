import logging
import pickle
import threading
from queue import Queue
from typing import Dict, Any, List, Optional, Callable

import grpc
from torch import Tensor

from core.dif_executor import DifJob
from core.ifr import IFR
from core.predictor import Predictor
from core.util import SerialTimer, timed_rpc
from rpc import msg_pb2_grpc
from rpc.msg_pb2 import Req, FinishMsg, IFRMsg, LayerCostMsg, PredictorsMsg


class AsyncClient(threading.Thread):
    """异步执行RPC，仅适用于不需要返回值的函数"""
    def __init__(self):
        super().__init__()
        self._que = Queue()
        self._logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> None:
        while True:
            func, args = self._que.get()
            self._logger.debug(f"{func.__name__}({args}) calling...")
            func(*args)
            self._logger.debug(f"{func.__name__}({args}) finished")

    def call_async(self, func: Callable, *args) -> None:
        self._que.put((func, args))
        self._logger.debug(f"{func.__name__}({args}) enqueued")


class MasterStub:
    def __init__(self, channel: grpc.Channel, aclient: AsyncClient):
        self._stub = msg_pb2_grpc.MasterStub(channel)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._client = aclient

    def report_finish(self, ifr_id: int, tensor4d: Tensor = None) -> None:
        self._client.call_async(self._report_finish, ifr_id, tensor4d)

    def _report_finish(self, ifr_id: int, tensor4d: Tensor = None) -> None:
        if tensor4d is None:
            msg = FinishMsg(ifr_id=ifr_id)
        else:
            with SerialTimer(SerialTimer.SType.DUMP, FinishMsg, self._logger):
                msg = FinishMsg(ifr_id=ifr_id, arr3d=DifJob.tensor4d_arr3dmsg(tensor4d))
        timed_rpc(self._stub.report_finish, msg, 'master', 's', self._logger)


class WorkerStub:
    def __init__(self, id_, channel: grpc.Channel, aclient: AsyncClient):
        self._name = f'worker{id_}'  # 请求目标的名称
        self._stub = msg_pb2_grpc.WorkerStub(channel)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._client = aclient

    def new_ifr(self, ifr: IFR) -> None:
        self._client.call_async(self._new_ifr, ifr)

    def _new_ifr(self, ifr: IFR) -> None:
        with SerialTimer(SerialTimer.SType.DUMP, IFRMsg, self._logger):
            msg = ifr.to_msg()
        timed_rpc(self._stub.new_ifr, msg, self._name, 's', self._logger)

    def layer_cost(self) -> List[float]:
        msg = timed_rpc(self._stub.layer_cost, Req(), self._name, 'r', self._logger)
        with SerialTimer(SerialTimer.SType.LOAD, LayerCostMsg, self._logger):
            costs = pickle.loads(msg.costs)
        return costs


class TrainerStub:
    def __init__(self, channel: grpc.Channel):
        self._stub = msg_pb2_grpc.TrainerStub(channel)
        self._logger = logging.getLogger(self.__class__.__name__)

    def get_predictors(self) -> List[Predictor]:
        msg = timed_rpc(self._stub.get_predictors, Req(), 'trainer', 'r', self._logger)
        with SerialTimer(SerialTimer.SType.LOAD, PredictorsMsg, self._logger):
            return pickle.loads(msg.predictors)


class MStubFactory:
    def __init__(self, config: Dict[str, Any]):
        wk_num = len(config['port']['worker'])
        net_config = config['net']
        self.wk_chan: List[Optional[grpc.Channel]] = [None for _ in range(wk_num)]
        for route, addr in net_config.items():
            if route.startswith('m->w'):
                self.wk_chan[int(route.replace('m->w', ''))] = grpc.insecure_channel(addr)
        assert all(chan is not None for chan in self.wk_chan)
        self.trn_chan = grpc.insecure_channel(net_config['m->t'])
        self.aclient = AsyncClient()
        self.aclient.start()

    def worker(self, worker_id: int) -> WorkerStub:
        return WorkerStub(worker_id, self.wk_chan[worker_id], self.aclient)

    def trainer(self) -> TrainerStub:
        return TrainerStub(self.trn_chan)


class WStubFactory:
    def __init__(self, id_: int, config: Dict[str, Any]):
        wk_num = len(config['port']['worker'])
        net_config = config['net']
        self.id = id_
        self.nwk_chan, self.mst_chan = None, None
        if id_ < wk_num - 1:
            self.nwk_chan = grpc.insecure_channel(net_config[f'w{id_}->w{id_+1}'])
        else:
            self.mst_chan = grpc.insecure_channel(net_config[f'w{id_}->m'])
        self.aclient = AsyncClient()
        self.aclient.start()

    def worker(self) -> WorkerStub:
        assert self.nwk_chan is not None
        return WorkerStub(self.id+1, self.nwk_chan, self.aclient)

    def master(self) -> MasterStub:
        assert self.mst_chan is not None
        return MasterStub(self.mst_chan, self.aclient)
