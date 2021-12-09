import logging
import pickle
from concurrent import futures
from typing import Any, Dict

import grpc

from core.util import SerialTimer
from rpc.msg_pb2 import Req, PredictorsMsg
from rpc import msg_pb2_grpc
from trainer.trainer import Trainer


class TrainerServicer(msg_pb2_grpc.TrainerServicer):
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.trainer = Trainer(config)
        self.trainer.start()
        self.__serve(str(config['port']['trainer']))

    def get_predictors(self, request: Req, context: grpc.ServicerContext) -> PredictorsMsg:
        predictors = self.trainer.get_predictors()
        with SerialTimer(SerialTimer.SType.DUMP, PredictorsMsg, self.logger):
            return PredictorsMsg(predictors=pickle.dumps(predictors))

    def __serve(self, port: str):
        MAX_MESSAGE_LENGTH = 1024*1024*1024   # 最大消息长度为1GB
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=5),
                             options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                      ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
        msg_pb2_grpc.add_TrainerServicer_to_server(self, server)
        server.add_insecure_port('[::]:' + port)
        server.start()
        self.logger.info("start serving...")
        server.wait_for_termination()
