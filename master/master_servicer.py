from concurrent import futures
from typing import Dict, Any

import grpc

from rpc import msg_pb2_grpc
from master.master import Master
from rpc.msg_pb2 import Rsp, ResultMsg
from rpc.stub_factory import StubFactory


class MasterServicer(msg_pb2_grpc.MasterServicer):
    def __init__(self, global_config: Dict[str, Any]):
        self.master = Master(StubFactory(global_config['addr']), global_config)
        self.master.start()
        self.__serve(global_config['addr']['master'].split(':')[1])

    def check_result(self, result_msg: ResultMsg, context: grpc.ServicerContext) -> Rsp:
        self.master.check_result(result_msg)
        return Rsp()

    def __serve(self, port: str):
        MAX_MESSAGE_LENGTH = 1024*1024*1024   # 最大消息长度为1GB
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=5),
                             options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                      ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
        msg_pb2_grpc.add_MasterServicer_to_server(self, server)
        server.add_insecure_port('[::]:' + port)
        server.start()
        server.wait_for_termination()
