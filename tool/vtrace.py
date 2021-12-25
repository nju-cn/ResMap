from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


@dataclass
class Event:
    ifr_id: int  # IFR号
    ifr_fin: bool  # 此IFR是否已完成
    act: str  # 事件名称
    timestamp: datetime  # 时间戳
    is_start: bool  # start还是finish


def read_events(filename: str) -> List[List[Event]]:
    events = []
    ifr_cnt = 0
    with open(filename, 'r') as file:
        for line in file:
            timestamp, is_start, act, ifr = line[:-1].split(' ')
            timestamp = datetime.fromisoformat(timestamp)
            is_start = (is_start == 'start')
            ifr_id = int(ifr[3:].replace('-finished', ''))
            ifr_fin = ('-finished' in ifr)
            ifr_cnt = max(ifr_cnt, ifr_id+1)
            events.append(Event(ifr_id, ifr_fin, act, timestamp, is_start))
    i_evts = [[] for _ in range(ifr_cnt)]
    for event in events:
        i_evts[event.ifr_id].append(event)
    return i_evts


@dataclass
class Stage:
    act: str  # 事件名称
    thread: str  # 此阶段所属的设备线程：m->, ->w0, w0, w0->, ->w1, w1, w1->, ...
    start: datetime  # 起始时间
    finish: datetime  # 结束时间


@dataclass
class IFRRecord:
    ifr_id: int
    start: datetime
    finish: datetime
    stages: List[Stage]


def read_ifr_records(m_tc: str, w_tcs: List[str], act2trd: Dict[Tuple[str, str], str]) -> List[IFRRecord]:
    mi_evts = read_events(m_tc)
    w_i_evts = [read_events(w_tc) for w_tc in w_tcs]
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
        d_name = ['m'] + [f'w{w}' for w in range(len(w_i_evts))]  # d->设备名
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
    trd2y = {t: i for i, t in enumerate(trds)}
    fig = plt.figure()
    ax = fig.subplots()
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
                plt.plot([x, x], [y, y+h], color='black')
    plt.show()


if __name__ == '__main__':
    NWORKER = 2

    TRD2ACTS = {'m->': ['encode', 'transmit']}
    for wid in range(NWORKER):
        TRD2ACTS[f'->w{wid}'] = ['decode']
        TRD2ACTS[f'w{wid}'] = ['execute']
        TRD2ACTS[f'w{wid}->'] = ['encode', 'transmit']
    ACT2TRD = {}  # (m, decode): 'm->', (w0, decode): '->w0'
    for trd, acts in TRD2ACTS.items():
        for act in acts:
            ACT2TRD[trd.replace('->', ''), act] = trd
    g_ircds = read_ifr_records('master.tc', [f'worker{i}.tc' for i in range(NWORKER)], ACT2TRD)
    for ircd in g_ircds:
        print(ircd)
    show_ifr_records(g_ircds, list(TRD2ACTS.keys()))
