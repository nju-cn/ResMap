import logging
from concurrent import futures
from typing import Dict, Any

import grpc

from core.util import SerialTimer, msg2tensor
from master.master import Master
from rpc import msg_pb2_grpc
from rpc.msg_pb2 import Rsp, FinishMsg
from rpc.stub_factory import MStubFactory


class MasterServicer(msg_pb2_grpc.MasterServicer):
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.master = Master(MStubFactory(config), config)
        self.master.start()
        self.__serve(str(config['port']['master']))

    def report_finish(self, finish_msg: FinishMsg, context: grpc.ServicerContext) -> Rsp:
        if len(finish_msg.arr3d.arr2ds) == 0:
            tensor = None
        else:
            with SerialTimer(SerialTimer.SType.LOAD, FinishMsg, self.logger):
                tensor = msg2tensor(finish_msg.arr3d)
        self.master.report_finish(finish_msg.ifr_id, tensor)
        return Rsp()

    def __serve(self, port: str):
        MAX_MESSAGE_LENGTH = 1024*1024*1024   # 最大消息长度为1GB
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=5),
                             options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                      ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
        msg_pb2_grpc.add_MasterServicer_to_server(self, server)
        server.add_insecure_port('[::]:' + port)
        server.start()
        self.logger.info("start serving...")
        server.wait_for_termination()
