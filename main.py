import logging.config
from typing import Dict, Any

import click
import humanfriendly.terminal
import yaml

from master.master_servicer import MasterServicer
from trainer.trainer_servicer import TrainerServicer
from worker.worker_servicer import WorkerServicer


def config_common(config_file: str) -> Dict[str, Any]:
    """设置启动LocalSimulator, MasterController, WorkerController前都需要配置的参数
    :param config_file 配置文件路径
    :return 从配置文件读取出的配置"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, yaml.Loader)  # 加载yaml中规定的类
    # 载入logger配置，因为可以加注释所以用了yaml格式，因为注释有中文所以读取编码限制为utf-8
    humanfriendly.terminal.enable_ansi_support()  # 增加对Windows彩色输出的支持
    with open(config['log_cfg'], 'r', encoding='utf-8') as f:
        logging.config.dictConfig(yaml.load(f, yaml.Loader))  # 载入logger配置
    return config


@click.group()
def start_service():
    """Choose master/worker/simulator to start"""


@start_service.command(name='master')
@click.option('-c', '--config', 'config_file', type=click.Path(),
              required=False, default='config.yml', show_default=True,
              help='Configuration file path of this project')
def start_master(config_file: str):
    """Start master with desired configuration"""
    config = config_common(config_file)
    MasterServicer(config)


@start_service.command(name='worker')
@click.option('-c', '--config', 'config_file', type=click.Path(),
              required=False, default='config.yml', show_default=True,
              help='Configuration file path of this project')
@click.option('-i', '--id', 'id_', type=int, required=True,
              help='The id of this worker, which should already be in the configuration file')
def start_worker(config_file: str, id_: int):
    """Start a worker with specified id and desired configuration"""
    config = config_common(config_file)
    WorkerServicer(id_, config)


@start_service.command(name='trainer')
@click.option('-c', '--config', 'config_file', type=click.Path(),
              required=False, default='config.yml', show_default=True,
              help='Configuration file path of this project')
def start_trainer(config_file):
    """Start a trainer with desired configuration"""
    config = config_common(config_file)
    TrainerServicer(config)


if __name__ == '__main__':
    start_service()
