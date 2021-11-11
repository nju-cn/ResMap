from typing import Dict, Any

import grpc

import msg_pb2_grpc


class StubFactory:
    def __init__(self, addr_config: Dict[str, Any]):
        self.mst_chan = grpc.insecure_channel(addr_config['master'])
        self.wk_chan = {wk: grpc.insecure_channel(addr) for wk, addr in addr_config['worker'].items()}

    def master(self) -> msg_pb2_grpc.MasterStub:
        return msg_pb2_grpc.MasterStub(self.mst_chan)

    def worker(self, worker_id: int) -> msg_pb2_grpc.WorkerStub:
        return msg_pb2_grpc.WorkerStub(self.wk_chan[worker_id])
