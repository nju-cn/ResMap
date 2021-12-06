import pickle
from typing import Dict, Any, List

import grpc
from torch import Tensor

from core.dif_executor import DifJob
from core.ifr import IFR
from core.predictor import Predictor
from rpc import msg_pb2_grpc
from rpc.msg_pb2 import Req, FinishMsg


class MasterStub:
    def __init__(self, channel: grpc.Channel):
        self._stub = msg_pb2_grpc.MasterStub(channel)

    def report_finish(self, ifr_id: int, tensor4d: Tensor = None) -> None:
        if tensor4d is None:
            self._stub.report_finish(FinishMsg(ifr_id=ifr_id))
        else:
            self._stub.report_finish(FinishMsg(ifr_id=ifr_id, arr3d=DifJob.tensor4d_arr3dmsg(tensor4d)))


class WorkerStub:
    def __init__(self, channel: grpc.Channel):
        self._stub = msg_pb2_grpc.WorkerStub(channel)

    def new_ifr(self, ifr: IFR) -> None:
        return self._stub.new_ifr(ifr.to_msg())

    def layer_cost(self) -> List[float]:
        return pickle.loads(self._stub.layer_cost(Req()).costs)


class TrainerStub:
    def __init__(self, channel: grpc.Channel):
        self._stub = msg_pb2_grpc.TrainerStub(channel)

    def get_predictors(self) -> List[Predictor]:
        return pickle.loads(self._stub.get_predictors(Req()).predictors)


class StubFactory:
    def __init__(self, addr_config: Dict[str, Any]):
        self.mst_chan = grpc.insecure_channel(addr_config['master'])
        self.wk_chan = {wk: grpc.insecure_channel(addr) for wk, addr in addr_config['worker'].items()}
        self.trn_chan = grpc.insecure_channel(addr_config['trainer'])

    def master(self) -> MasterStub:
        return MasterStub(self.mst_chan)

    def worker(self, worker_id: int) -> WorkerStub:
        return WorkerStub(self.wk_chan[worker_id])

    def trainer(self) -> TrainerStub:
        return TrainerStub(self.trn_chan)
