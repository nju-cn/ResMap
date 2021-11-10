from concurrent import futures
from typing import Dict, Any

import grpc

from msg_pb2 import IFRMsg, Rsp, ResultMsg
import msg_pb2_grpc
from worker import Worker


class WorkerServicer(msg_pb2_grpc.WorkerServicer):
    def __init__(self, worker_id: int, config_dict: Dict[str, Any]):
        self.worker = Worker(worker_id, config_dict['dnn_loader'], config_dict['check'],
                             self.__send_ifr_async, self.__check_result)
        self.mst_addr = config_dict['addr']['master']
        self.wk_addr = config_dict['addr']['worker']
        self.worker.start()
        self.__serve()

    def new_ifr(self, ifr_msg: IFRMsg, context: grpc.ServicerContext) -> Rsp:
        self.worker.new_ifr(ifr_msg)
        return Rsp()

    def __send_ifr_async(self, ifr_msg: IFRMsg) -> None:
        channel = grpc.insecure_channel(self.wk_addr[ifr_msg.worker_id])
        stub = msg_pb2_grpc.WorkerStub(channel)
        stub.new_ifr(ifr_msg)

    def __check_result(self, result_msg: ResultMsg) -> None:
        channel = grpc.insecure_channel(self.mst_addr)
        stub = msg_pb2_grpc.MasterStub(channel)
        stub.check_result(result_msg)

    def __serve(self):
        MAX_MESSAGE_LENGTH = 1024*1024*1024   # 最大消息长度为1GB
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=5),
                             options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                      ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
        msg_pb2_grpc.add_WorkerServicer_to_server(self, server)
        server.add_insecure_port('[::]:'+self.wk_addr[self.worker.id()].split(':')[1])
        server.start()
        server.wait_for_termination()
