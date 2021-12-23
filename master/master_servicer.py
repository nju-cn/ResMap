import logging
import threading
from typing import Dict, Any

import grpc

from core.raw_dnn import RawDNN
from core.util import SerialTimer, msg2tensor
from master.master import Master
from rpc import msg_pb2_grpc
from rpc.msg_pb2 import FinishMsg, Req
from rpc.stub_factory import MStubFactory


class MasterServicer:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.wk_addr = ['' for _ in range(len(config['port']['worker']))]
        for rt, addr in config['net'].items():
            if rt.startswith('m->w'):
                self.wk_addr[int(rt.replace('m->w', ''))] = addr
        self.master = Master(len(config['port']['worker']), RawDNN(config['dnn_loader']()),
                             config['video_path'], config['frame_size'], config['job'], config['check'],
                             MStubFactory(config), config['master'])
        for addr in self.wk_addr:
            threading.Thread(target=self.__report_finish_rev, args=(addr,)).start()
        self.master.start()

    def __report_finish_rev(self, addr: str) -> None:
        while 1:
            stub = msg_pb2_grpc.WorkerStub(grpc.insecure_channel(addr))
            response = stub.report_finish_rev(Req())
            for finish_msg in response:
                self.__report_finish(finish_msg)

    def __report_finish(self, finish_msg: FinishMsg) -> None:
        self.logger.info(f"finish IFR{finish_msg.ifr_id}", extra={'trace': True})
        if len(finish_msg.arr3d.arr2ds) == 0:
            tensor = None
        else:
            with SerialTimer(SerialTimer.SType.LOAD, FinishMsg, self.logger):
                tensor = msg2tensor(finish_msg.arr3d)
        self.master.report_finish(finish_msg.ifr_id, tensor)
