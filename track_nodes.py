#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_nodes.py — 绕过 ExaDiS 的 tag 回收,按"位置连续性"追踪节点跨帧行为。

为什么需要:.data 里那个"tag"其实是节点在数组里的下标(write_data 写的是循环变量 i),
每次 purge_network 压缩数组后下标重排,看起来就像 tag 被回收/乱跳。本脚本不信下标,
改用"相邻帧最近位置匹配"把同一个物理节点连成轨迹。依据:每帧位移很小(~10-40),
远小于节点间距(~300-600),就近匹配几乎无歧义。轨迹在拓扑事件(合并/删除/释放/分裂)
处会自然中断——这恰好告诉我们节点是什么时候、在哪儿被创建或销毁的。

用法:
  # 1) 先列出某帧的节点,挑一个起点(可只看 c9)
  python track_nodes.py <数据目录> --list 928 --c9

  # 2) 追踪 928 帧里最靠近 (6471,5973,5665) 的那个节点,逐帧打到 960 帧
  python track_nodes.py <数据目录> --near 6471 5973 5665 --from 928 --to 960

  # move = 本帧相对上帧的位移大小;dir=<-REV 表示位移方向相对上一帧反转(=抽搐指纹)
  # --maxmove 控制"还算同一个节点"的最大跳距(默认150);超了就判轨迹中断
"""
import sys, os, glob, re, math, argparse
try:
    sys.stdout.reconfigure(encoding='utf-8')   # 修 Windows 控制台中文乱码
except Exception:
    pass

PRIM = re.compile(r'^(\d+),\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s*$')

def parse_frame(path):
    nodes = []
    with open(path) as f:
        for line in f:
            m = PRIM.match(line)
            if m:
                nodes.append(dict(idx=int(m.group(2)),
                                  x=float(m.group(3)), y=float(m.group(4)), z=float(m.group(5)),
                                  arms=int(m.group(6)), c=int(m.group(7))))
    return nodes

def frame_files(datadir, f_from=None, f_to=None):
    out = []
    for f in glob.glob(os.path.join(datadir, 'config.*.data')):
        m = re.search(r'config\.(\d+)\.data$', f.replace('\\', '/'))
        if not m:
            continue
        fr = int(m.group(1))
        if f_from is not None and fr < f_from: continue
        if f_to   is not None and fr > f_to:   continue
        out.append((fr, f))
    out.sort()
    return out

def dist(a, b):
    return math.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)

def nearest(nodes, x, y, z, c9only=False):
    t = dict(x=x, y=y, z=z); best, bd = None, 1e30
    for n in nodes:
        if c9only and n['c'] != 9: continue
        d = dist(n, t)
        if d < bd: bd, best = d, n
    return best, bd

def cmd_list(datadir, fr, c9only):
    fs = frame_files(datadir, fr, fr)
    if not fs:
        print("找不到该帧"); return
    nodes = parse_frame(fs[0][1])
    print(f"# frame {fr}: {len(nodes)} 个节点" + ("(只列 c9)" if c9only else ""))
    print(f"{'x':>8} {'y':>8} {'z':>8} {'c':>3} {'arms':>4}")
    for n in sorted(nodes, key=lambda n:(n['x'],n['y'],n['z'])):
        if c9only and n['c'] != 9: continue
        print(f"{n['x']:8.0f} {n['y']:8.0f} {n['z']:8.0f} {n['c']:3d} {n['arms']:4d}")

def cmd_track(datadir, x, y, z, f_from, f_to, maxmove):
    fs = frame_files(datadir, f_from, f_to)
    if not fs:
        print("没有匹配的帧"); return
    fr0, p0 = fs[0]
    cur, d0 = nearest(parse_frame(p0), x, y, z)
    if cur is None:
        print("起始帧无节点"); return
    print(f"# 起点 frame {fr0}: 最近节点距目标 {d0:.0f}  (maxmove={maxmove})")
    print(f"{'frame':>6} {'x':>8} {'y':>8} {'z':>8} {'c':>3} {'arms':>4} {'move':>7} {'dir':>6}")
    prev = (cur['x'], cur['y'], cur['z']); pdisp = None
    print(f"{fr0:6d} {cur['x']:8.0f} {cur['y']:8.0f} {cur['z']:8.0f} {cur['c']:3d} {cur['arms']:4d} {'-':>7} {'-':>6}")
    nrev = 0; nfroz = 0
    for fr, p in fs[1:]:
        nxt, dd = nearest(parse_frame(p), *prev)
        if nxt is None or dd > maxmove:
            print(f"{fr:6d}  >>> 轨迹中断(最近 {dd:.0f} > maxmove)= 该节点被合并/删除/释放,或一步大跳")
            break
        disp = (nxt['x']-prev[0], nxt['y']-prev[1], nxt['z']-prev[2])
        flag = ''
        if pdisp is not None and dd > 0.5 and (disp[0]*pdisp[0]+disp[1]*pdisp[1]+disp[2]*pdisp[2]) < 0:
            flag = '<-REV'; nrev += 1   # 只在真有位移(>0.5)时算反向,滤掉亚像素抖动噪声
        if dd < 0.5: nfroz += 1
        print(f"{fr:6d} {nxt['x']:8.0f} {nxt['y']:8.0f} {nxt['z']:8.0f} {nxt['c']:3d} {nxt['arms']:4d} {dd:7.1f} {flag:>6}")
        prev = (nxt['x'], nxt['y'], nxt['z']); pdisp = disp
    print(f"# {nrev} 次真实方向反转(抽搐);{nfroz} 帧几乎不动(move<0.5,=卡住/冻结)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datadir')
    ap.add_argument('--list', type=int, metavar='FRAME')
    ap.add_argument('--c9', action='store_true')
    ap.add_argument('--near', nargs=3, type=float, metavar=('X','Y','Z'))
    ap.add_argument('--from', dest='f_from', type=int)
    ap.add_argument('--to', dest='f_to', type=int)
    ap.add_argument('--maxmove', type=float, default=150.0)
    a = ap.parse_args()
    if a.list is not None:
        cmd_list(a.datadir, a.list, a.c9)
    elif a.near is not None and a.f_from is not None:
        cmd_track(a.datadir, a.near[0], a.near[1], a.near[2], a.f_from, a.f_to, a.maxmove)
    else:
        ap.print_help()

if __name__ == '__main__':
    main()
