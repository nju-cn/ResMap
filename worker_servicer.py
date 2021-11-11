from concurrent import futures
from typing import Dict, Any

import grpc

from msg_pb2 import IFRMsg, Rsp
import msg_pb2_grpc
from worker import Worker
from stub_factory import StubFactory


class WorkerServicer(msg_pb2_grpc.WorkerServicer):
    def __init__(self, worker_id: int, global_config: Dict[str, Any]):
        self.worker = Worker(worker_id, global_config['dnn_loader'], global_config['check'],
                             StubFactory(global_config['addr']))
        self.worker.start()
        self.__serve(global_config['addr']['worker'][worker_id].split(':')[1])

    def new_ifr(self, ifr_msg: IFRMsg, context: grpc.ServicerContext) -> Rsp:
        self.worker.new_ifr(ifr_msg)
        return Rsp()

    def __serve(self, port: str):
        MAX_MESSAGE_LENGTH = 1024*1024*1024   # 最大消息长度为1GB
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=5),
                             options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                      ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
        msg_pb2_grpc.add_WorkerServicer_to_server(self, server)
        server.add_insecure_port('[::]:' + port)
        server.start()
        server.wait_for_termination()
