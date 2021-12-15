import logging
import pickle
from queue import Queue
from concurrent import futures
from typing import Dict, Any

import grpc

from core.ifr import IFR
from core.raw_dnn import RawDNN
from core.util import SerialTimer
from rpc.msg_pb2 import IFRMsg, Rsp, Req, LayerCostMsg, FinishMsg
from rpc import msg_pb2_grpc
from rpc.stub_factory import WStubFactory
from worker.worker import Worker


class WorkerServicer(msg_pb2_grpc.WorkerServicer):
    def __init__(self, worker_id: int, config: Dict[str, Any]):
        self.job_type = config['job']
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rev_que = Queue()
        self.worker = Worker(worker_id, RawDNN(config['dnn_loader']()), config['frame_size'], config['check'],
                             config['executor'], WStubFactory(worker_id, self.rev_que, config), config['worker'])
        self.worker.start()
        self.__serve(str(config['port']['worker'][worker_id]))

    def new_ifr(self, ifr_msg: IFRMsg, context: grpc.ServicerContext) -> Rsp:
        with SerialTimer(SerialTimer.SType.LOAD, IFRMsg, self.logger):
            ifr = IFR.from_msg(ifr_msg, self.job_type)
        self.worker.new_ifr(ifr)
        return Rsp()

    def layer_cost(self, req: Req, context: grpc.ServicerContext) -> LayerCostMsg:
        costs = self.worker.layer_cost()
        with SerialTimer(SerialTimer.SType.DUMP, LayerCostMsg, self.logger):
            return LayerCostMsg(costs=pickle.dumps(costs))

    def report_finish_rev(self, req: Req, context: grpc.ServicerContext) -> FinishMsg:
        while 1:
            yield self.rev_que.get()

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
