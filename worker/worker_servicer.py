import logging
import pickle
from concurrent import futures
from typing import Dict, Any

import grpc

from core.ifr import IFR
from rpc.msg_pb2 import IFRMsg, Rsp, Req, LayerCostMsg
from rpc import msg_pb2_grpc
from worker.worker import Worker
from rpc.stub_factory import StubFactory


class WorkerServicer(msg_pb2_grpc.WorkerServicer):
    def __init__(self, worker_id: int, global_config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.worker = Worker(worker_id, StubFactory(global_config['addr']), global_config)
        self.worker.start()
        self.__serve(global_config['addr']['worker'][worker_id].split(':')[1])

    def new_ifr(self, ifr_msg: IFRMsg, context: grpc.ServicerContext) -> Rsp:
        self.worker.new_ifr(IFR.from_msg(ifr_msg))
        return Rsp()

    def layer_cost(self, req: Req, context: grpc.ServicerContext) -> LayerCostMsg:
        return LayerCostMsg(costs=pickle.dumps(self.worker.layer_cost()))

    def __serve(self, port: str):
        MAX_MESSAGE_LENGTH = 1024*1024*1024   # 最大消息长度为1GB
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=5),
                             options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                      ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
        msg_pb2_grpc.add_WorkerServicer_to_server(self, server)
        server.add_insecure_port('[::]:' + port)
        server.start()
        self.logger.info("start serving...")
        server.wait_for_termination()
