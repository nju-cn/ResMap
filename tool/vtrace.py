import glob
import os
import shutil
import zipfile
from datetime import datetime
from dataclasses import dataclass
from io import TextIOWrapper
from typing import List, Dict, Tuple, Optional, TextIO, Union

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import paramiko
import yaml


@dataclass
class Event:
    ifr_id: int  # IFR号
    ifr_fin: bool  # 此IFR是否已完成
    act: str  # 事件名称
    timestamp: datetime  # 时间戳
    is_start: bool  # start还是finish


def read_events(tcfile: Union[str, TextIO], ifr_num: int = -1) -> List[List[Event]]:
    """从某个设备生成的tc文件中读取事件
    :param tcfile tc文件名，或者file-like对象
    :param ifr_num 所有事件中的IFR数量(id从0计数到ifr_num-1)。若未知可不填
    :return i_evts i_evts[i]对应id=i的IFR所有事件，事件按照时间顺序排序
    """
    events = []
    ifr_cnt = 0
    if isinstance(tcfile, str):
        tcfile = open(tcfile, 'r')
    with tcfile:
        for line in tcfile:
            timestamp, is_start, act, ifr = line[:-1].split(' ')
            timestamp = datetime.fromisoformat(timestamp)
            is_start = (is_start == 'start')
            ifr_id = int(ifr[3:].replace('-finished', ''))
            ifr_fin = ('-finished' in ifr)
            ifr_cnt = max(ifr_cnt, ifr_id+1)
            events.append(Event(ifr_id, ifr_fin, act, timestamp, is_start))
    i_evts = [[] for _ in range(max(ifr_cnt, ifr_num))]
    for event in events:
        i_evts[event.ifr_id].append(event)
    return i_evts


@dataclass
class Stage:
    act: str  # 事件名称
    thread: str  # 此阶段所属的设备线程：m->, ->w0, w0, w0->, ->w1, w1, w1->, ...
    start: datetime  # 起始时间
    finish: datetime  # 结束时间

    def __repr__(self):
        return f"Stage('{self.trans_thread(self.thread)}':{self.act}," \
               f"s='{self.start.isoformat(' ', timespec='milliseconds')}'," \
               f"f='{self.finish.isoformat(' ', timespec='milliseconds')}')"

    @staticmethod
    def trans_thread(thread: str):
        return thread.replace('$', '').replace(r'\rightarrow', '->').replace('_', '').replace(' ', '')


@dataclass
class IFRRecord:
    ifr_id: int
    start: datetime
    finish: datetime
    stages: List[Stage]

    def __str__(self):
        return f"IFRRecord(ifr={self.ifr_id}, start='{self.start.isoformat(' ', timespec='milliseconds')}', " \
               f"finish='{self.finish.isoformat(' ', timespec='milliseconds')}', " \
               f"stages={self.stages})"


def events2records(mi_evts: List[List[Event]], w_i_evts: List[List[List[Event]]],
                   act2trd: Dict[Tuple[str, str], str]) -> List[IFRRecord]:
    ircds = []
    for ifr_id in range(len(mi_evts)):
        m_evts = mi_evts[ifr_id]  # Master中当前IFR的所有事件
        # 从Master的事件中找到当前IFR的开始结束时间，并删除该事件
        se, fe = -1, -1
        for e, evt in enumerate(m_evts):
            if evt.act == 'process':
                if evt.is_start:
                    assert se < 0
                    se = e
                else:
                    assert fe < 0
                    fe = e
        start, finish = m_evts[se].timestamp, m_evts[fe].timestamp
        m_evts.pop(fe)  # 这里要先删除后面的finish事件，这样se的索引不会变
        m_evts.pop(se)
        # 根据事件，生成Stage
        ircd = IFRRecord(ifr_id, start, finish, [])
        d_evts = [m_evts] + [(i_evts[ifr_id] if i_evts else []) for i_evts in w_i_evts]  # 设备->当前IFR的所有事件
        d_name = ['$m$'] + [f'$w_{w}$' for w in range(len(w_i_evts))]  # d->设备名
        tr_sevt: Optional[Event] = None  # 最近一个传输start事件
        tr_sd: int = -1  # 最近一个传输start事件对应的设备
        for d, evts in enumerate(d_evts):  # 遍历各设备
            if d == 0:  # Master至少有一个事件，且第一个事件应该是start
                assert len(evts) > 0 and evts[0].is_start
                e = 0
            else:
                if len(evts) == 0:  # Worker上一个事件也没有，直接跳过
                    continue
                # 如果Worker上有事件，第一个事件应该是传输完成事件
                assert evts[0].act == 'transmit' and not evts[0].is_start
                s_evt, f_evt = tr_sevt, evts[0]
                ircd.stages.append(Stage(s_evt.act, act2trd[d_name[tr_sd], s_evt.act],
                                         s_evt.timestamp, f_evt.timestamp))
                e = 1
            while e+1 < len(evts):
                s_evt, f_evt = evts[e], evts[e+1]
                ircd.stages.append(Stage(s_evt.act, act2trd[d_name[d], s_evt.act], s_evt.timestamp, f_evt.timestamp))
                e += 2
            assert e+1 == len(evts)  # len(evts)-1应该是传输开始事件
            assert evts[-1].act == 'transmit' and evts[-1].is_start
            tr_sevt = evts[-1]
            tr_sd = d
        # 最后一个传输事件的finish时间为此IFR的finish时间，即Master收到的时间
        ircd.stages.append(Stage(tr_sevt.act, act2trd[d_name[tr_sd], tr_sevt.act], tr_sevt.timestamp, ircd.finish))
        ircds.append(ircd)
    return ircds


def show_ifr_records(ifr_records: List[IFRRecord], trds: List[str]):
    """trds为各设备的线程，按照执行顺序排列"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    lg = {'size': 16}

    trd2y = {t: i for i, t in enumerate(trds)}
    fig = plt.figure()
    ax = fig.subplots()
    plt.tick_params(labelsize=13)
    ax.set_xlabel('时间(s)', fontproperties=lg)
    ax.invert_yaxis()
    plt.yticks(list(range(len(trds))), trds)
    colors = list(mcolors.XKCD_COLORS.values())
    all_start = ifr_records[0].start  # 最开始的时间
    for ifr_rcd in ifr_records:
        color = colors[ifr_rcd.ifr_id]
        for stage in ifr_rcd.stages:
            bar, = plt.barh(trd2y[stage.thread], (stage.finish-stage.start).total_seconds(),
                           left=(stage.start-all_start).total_seconds(), color=color)
            # 在encode和transmit之间画一条分界线
            if stage.act == 'transmit':
                x, y = bar.get_xy()
                w, h = bar.get_width(), bar.get_height()
                plt.plot([x, x], [y, y+h], color='black', linewidth=.7)
    plt.tight_layout()
    plt.show()


def read_from_zip(zip_name: str) -> Tuple[List[List[Event]], List[List[List[Event]]]]:
    """从指定的zip文件中读取mi_evts, w_i_evts"""
    with zipfile.ZipFile(zip_name) as tczip:
        nworker = 1 + max(int(tcname.replace('worker', '').replace('.tc', ''))
                          for tcname in tczip.namelist() if tcname.startswith('worker'))
        with tczip.open('master.tc') as m_tcfile:
            mi_evts = read_events(TextIOWrapper(m_tcfile))
        w_i_evts = []
        for wk in range(nworker):
            with tczip.open(f'worker{wk}.tc') as w_tcfile:
                w_i_evts.append(read_events(TextIOWrapper(w_tcfile), len(mi_evts)))
        return mi_evts, w_i_evts


def read_from_remote(device_cfg: str) -> Tuple[List[List[Event]], List[List[List[Event]]]]:
    """根据device_cfg指定的device.yml，从远程服务器获取tc文件，并读取mi_evts, w_i_evts"""
    CACHE_DIR = 'tc-cache/'
    REMOTE_DIR = 'cnn-video/'

    print("downloading tc from devices...")
    if os.path.isdir(CACHE_DIR):
        # 删除缓存目录，以确保缓存目录下都是最新的tc文件
        shutil.rmtree(CACHE_DIR)
    # 创建新的缓存目录
    os.mkdir(CACHE_DIR)
    with open(device_cfg, 'r', encoding='utf-8') as f:
        config = yaml.load(f, yaml.Loader)
    for role in config['role']:
        devs = config['role'][role] if role == 'w' else [config['role']['m']]
        for d, dev_name in enumerate(devs):
            username = config['device'][dev_name]['user']
            ip, port = config['device'][dev_name]['addr'].split(':')
            tran = paramiko.Transport((ip, int(port)))
            tran.connect(username=username, password=config['user'][username])
            sftp = paramiko.SFTPClient.from_transport(tran)
            remotedir = '/root/' if username == 'root' else f'/home/{username}/'
            remotedir += REMOTE_DIR
            filename = 'master.tc' if role == 'm' else f'worker{d}.tc'
            sftp.get(remotedir + filename, CACHE_DIR + filename)
    nworker = len(config['role']['w'])
    mi_evts = read_events(f'{CACHE_DIR}/master.tc')
    w_i_evts = [read_events(f'{CACHE_DIR}/worker{i}.tc', len(mi_evts)) for i in range(nworker)]
    return mi_evts, w_i_evts


def read_from_local(local_dir: str) -> Tuple[List[List[Event]], List[List[List[Event]]]]:
    """从指定的local_dir目录中寻找tc文件，读取mi_evts, w_i_evts"""
    nworker = len(glob.glob(f'{local_dir}/worker*.tc'))  # 自动识别worker数量
    mi_evts = read_events(f'{local_dir}/master.tc')
    w_i_evts = [read_events(f'{local_dir}/worker{i}.tc', len(mi_evts)) for i in range(nworker)]
    return mi_evts, w_i_evts


if __name__ == '__main__':
    # 3种模式：
    #   l: local模式，从 LOCAL_DIR 指定的目录下寻找tc文件，根据目录下的文件名判断worker数
    #   r: remote模式，从 REMOTE_CFG 获取远程服务器配置和worker数，下载远程tc文件
    #   z: zip模式，从 TCZIP 指定的zip压缩包中读取tc文件，根据压缩包中的文件名判断worker数
    MODE = 'r'

    LOCAL_DIR = 'lbc2'  # l模式下, 本地目录路径
    REMOTE_CFG = 'device.yml'  # 远程服务器的配置文件
    TCZIP = 'edge.zip'  # 从zip文件中读取tc文件

    if MODE == 'l':
        g_mi_evts, g_w_i_evts = read_from_local(LOCAL_DIR)
    elif MODE == 'r':
        g_mi_evts, g_w_i_evts = read_from_remote(REMOTE_CFG)
    elif MODE == 'z':
        g_mi_evts, g_w_i_evts = read_from_zip(TCZIP)
    else:
        raise Exception(f"No such mode '{MODE}'")
    print(f"events read succeeded, n_worker={len(g_w_i_evts)}, n_ifr={len(g_mi_evts)}")

    TRD2ACTS = {r'$m\rightarrow$': ['encode', 'transmit']}
    for wid in range(len(g_w_i_evts)):
        TRD2ACTS[rf'$\rightarrow w_{wid}$'] = ['decode']
        TRD2ACTS[f'$w_{wid}$'] = ['execute']
        TRD2ACTS[rf'$w_{wid}\rightarrow$'] = ['encode', 'transmit']
    ACT2TRD = {}  # (m, decode): 'm->', (w0, decode): '->w0'
    for trd, acts in TRD2ACTS.items():
        for act in acts:
            ACT2TRD[trd.replace(r'\rightarrow', '').replace(' ', ''), act] = trd

    print("transforming and ploting...")
    g_ircds = events2records(g_mi_evts, g_w_i_evts, ACT2TRD)
    for ircd in g_ircds:
        print(ircd)
    show_ifr_records(g_ircds, list(TRD2ACTS.keys()))
