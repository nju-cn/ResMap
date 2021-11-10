from concurrent import futures
from typing import Dict, Any

import grpc

import msg_pb2_grpc
from master import Master
from msg_pb2 import IFRMsg, Rsp, Arr3dMsg, ResultMsg


class MasterServicer(msg_pb2_grpc.MasterServicer):
    def __init__(self, config_dict: Dict[str, Any]):
        self.master = Master(config_dict['dnn_loader'], config_dict['video_path'],
                             config_dict['frame_size'], config_dict['check'],
                             self.__send_ifr_async, config_dict.get('master', {}))
        self.mst_addr = config_dict['addr']['master']
        self.wk_addr = config_dict['addr']['worker']
        self.master.start()
        self.__serve()

    def check_result(self, result_msg: ResultMsg, context: grpc.ServicerContext) -> Rsp:
        self.master.check_result(result_msg)
        return Rsp()

    def __send_ifr_async(self, ifr_msg: IFRMsg) -> None:
        channel = grpc.insecure_channel(self.wk_addr[ifr_msg.wk_jobs[0].worker_id])
        stub = msg_pb2_grpc.WorkerStub(channel)
        stub.new_ifr(ifr_msg)

    def __serve(self):
        MAX_MESSAGE_LENGTH = 1024*1024*1024   # 最大消息长度为1GB
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=5),
                             options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                      ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
        msg_pb2_grpc.add_MasterServicer_to_server(self, server)
        server.add_insecure_port('[::]:'+self.mst_addr.split(':')[1])
        server.start()
        server.wait_for_termination()
