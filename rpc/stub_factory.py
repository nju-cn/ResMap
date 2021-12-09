import logging
import pickle
from typing import Dict, Any, List, Optional

import grpc
from torch import Tensor

from core.dif_executor import DifJob
from core.ifr import IFR
from core.predictor import Predictor
from core.util import SerialTimer, timed_rpc
from rpc import msg_pb2_grpc
from rpc.msg_pb2 import Req, FinishMsg, IFRMsg, LayerCostMsg, PredictorsMsg


class MasterStub:
    def __init__(self, channel: grpc.Channel):
        self._stub = msg_pb2_grpc.MasterStub(channel)
        self._logger = logging.getLogger(self.__class__.__name__)

    def report_finish(self, ifr_id: int, tensor4d: Tensor = None) -> None:
        if tensor4d is None:
            msg = FinishMsg(ifr_id=ifr_id)
        else:
            with SerialTimer(SerialTimer.SType.DUMP, FinishMsg, self._logger):
                msg = FinishMsg(ifr_id=ifr_id, arr3d=DifJob.tensor4d_arr3dmsg(tensor4d))
        timed_rpc(self._stub.report_finish, msg, 'master', 's', self._logger)


class WorkerStub:
    def __init__(self, id_, channel: grpc.Channel):
        self._name = f'worker{id_}'  # 请求目标的名称
        self._stub = msg_pb2_grpc.WorkerStub(channel)
        self._logger = logging.getLogger(self.__class__.__name__)

    def new_ifr(self, ifr: IFR) -> None:
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

    def worker(self, worker_id: int) -> WorkerStub:
        return WorkerStub(worker_id, self.wk_chan[worker_id])

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

    def worker(self) -> WorkerStub:
        assert self.nwk_chan is not None
        return WorkerStub(self.id+1, self.nwk_chan)

    def master(self) -> MasterStub:
        assert self.mst_chan is not None
        return MasterStub(self.mst_chan)
