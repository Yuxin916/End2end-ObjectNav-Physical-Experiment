#!/usr/bin/env python3
"""
profiling_report.py — aggregate real-time ObjectNav timing logs into the CORL
appendix table.

Usage:
    python3 scripts/profiling_report.py LOG_FILE [LOG_FILE ...]

Pass every log that may contain the markers (e.g. the vlm_navigator console
capture, its target_debug_*.log, and the sam2_detector console capture). Lines
are matched by their `[marker]` tag regardless of the ROS log prefix, so mixing
files is fine. `--episodes N` records how many episodes the logs span (default
inferred from the number of 'new_goal' triggers, min 1).

This script does NOT touch the robot — run it offline after an episode.
"""
import argparse
import re
import sys
from statistics import mean, pstdev


def _f(stats):
    """mean ± std formatter for a list of floats (ms or unitless)."""
    if not stats:
        return 'n/a'
    if len(stats) == 1:
        return f'{stats[0]:.1f} ± 0.0  (n=1)'
    return f'{mean(stats):.1f} ± {pstdev(stats):.1f}  (n={len(stats)})'


def _f2(stats):
    if not stats:
        return 'n/a'
    if len(stats) == 1:
        return f'{stats[0]:.2f}  (n=1)'
    return f'{mean(stats):.2f} ± {pstdev(stats):.2f}  (n={len(stats)})'


# Regexes for each marker. They search anywhere in the line.
RE = {
    'pipeline': re.compile(
        r'\[pipeline_ms\]\s+bev_update=(\S+)\s+frontier=(\S+)\s+render=(\S+)\s+'
        r'vlm=(\S+)\s+total=(\S+)\s+candidates=(\S+)(?:\s+gpu_mem=(\S+?)GB)?'),
    'bev': re.compile(r'\[bev_update_ms\]\s+(\S+)'),
    'detector': re.compile(r'\[detector_ms\]\s+(\S+)'),
    'cam': re.compile(r'\[cam_timing ms\].*?\btotal=(\S+)'),
    'cam_proj': re.compile(r'\[cam_timing ms\]\s+decode=(\S+)\s+project=(\S+)'),
    'lidar': re.compile(r'\[lidar_hz\]\s+avg_interval=(\S+?)s\s+freq=(\S+?)Hz'),
    'speed': re.compile(r'\[robot_speed\]\s+avg=(\S+?)m/s'),
    'wp_exec': re.compile(r'\[wp_exec_s\]\s+(\S+?)s\s+dist=(\S+?)m'),
    'rq': re.compile(
        r'\[re_query_stats\]\s+periodic=(\d+)\s+waypoint=(\d+)\s+'
        r'detection=(\d+)\s+new_goal=(\d+)\s+avg_interval=(\S+?)s'),
    'gpu_peak': re.compile(r'GPU memory:\s+peak=(\S+?)GB\s+/\s+total=(\S+?)GB'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--episodes', type=int, default=None)
    args = ap.parse_args()

    bev, frontier, render, vlm, total, cand, gpu_cur = [], [], [], [], [], [], []
    bev_solo, detector, cam_total, cam_proj = [], [], [], []
    lidar_hz, speed, wp_exec, wp_dist = [], [], [], []
    gpu_peak = gpu_total = None
    rq_last = None  # latest cumulative re_query_stats tuple

    for path in args.logs:
        try:
            with open(path, errors='replace') as fh:
                for line in fh:
                    m = RE['pipeline'].search(line)
                    if m:
                        g = m.groups()
                        for lst, idx in ((bev, 0), (frontier, 1), (render, 2),
                                         (vlm, 3), (total, 4)):
                            try:
                                v = float(g[idx])
                                if v >= 0:
                                    lst.append(v)
                            except ValueError:
                                pass
                        try:
                            cand.append(float(g[5]))
                        except ValueError:
                            pass
                        if g[6] is not None:
                            try:
                                v = float(g[6])
                                if v >= 0:
                                    gpu_cur.append(v)
                            except ValueError:
                                pass
                        continue
                    m = RE['bev'].search(line)
                    if m:
                        try:
                            bev_solo.append(float(m.group(1)))
                        except ValueError:
                            pass
                        continue
                    m = RE['detector'].search(line)
                    if m:
                        try:
                            detector.append(float(m.group(1)))
                        except ValueError:
                            pass
                        continue
                    m = RE['cam'].search(line)
                    if m:
                        try:
                            cam_total.append(float(m.group(1)))
                        except ValueError:
                            pass
                        mp = RE['cam_proj'].search(line)
                        if mp:
                            try:
                                cam_proj.append(float(mp.group(1)) + float(mp.group(2)))
                            except ValueError:
                                pass
                        continue
                    m = RE['lidar'].search(line)
                    if m:
                        try:
                            lidar_hz.append(float(m.group(2)))
                        except ValueError:
                            pass
                        continue
                    m = RE['speed'].search(line)
                    if m:
                        try:
                            speed.append(float(m.group(1)))
                        except ValueError:
                            pass
                        continue
                    m = RE['wp_exec'].search(line)
                    if m:
                        try:
                            wp_exec.append(float(m.group(1)))
                            wp_dist.append(float(m.group(2)))
                        except ValueError:
                            pass
                        continue
                    m = RE['rq'].search(line)
                    if m:
                        rq_last = tuple(int(x) for x in m.groups()[:4])
                        continue
                    m = RE['gpu_peak'].search(line)
                    if m:
                        gpu_peak, gpu_total = float(m.group(1)), float(m.group(2))
                        continue
        except FileNotFoundError:
            print(f'WARNING: log not found: {path}', file=sys.stderr)

    n_cycles = len(total) if total else len(vlm)
    if args.episodes is not None:
        episodes = args.episodes
    else:
        episodes = max(1, rq_last[3] if rq_last else 1)

    # Use the solo [bev_update_ms] stream if pipeline column was sparse.
    bev_report = bev if len(bev) >= len(bev_solo) else bev_solo

    print('=' * 64)
    print(' Real-time ObjectNav — latency profiling report')
    print('=' * 64)
    print(f'{"BEV map update latency (ms)":42} {_f(bev_report)}')
    print(f'{"Frontier extraction latency (ms)":42} {_f(frontier)}')
    print(f'{"BEV RGB rendering latency (ms)":42} {_f(render)}')
    print(f'{"VLM inference latency (ms)":42} {_f(vlm)}')
    print(f'{"Total decision cycle (ms)":42} {_f(total)}')
    print(f'{"Target detector latency (ms, parallel)":42} {_f(detector)}')
    print(f'{"Camera decode+proj (ms, parallel)":42} {_f(cam_proj)}')
    print(f'{"Camera total (ms, parallel)":42} {_f(cam_total)}')
    if gpu_peak is not None:
        print(f'{"Peak GPU memory (GB)":42} {gpu_peak:.2f} / {gpu_total:.2f}')
    else:
        print(f'{"Peak GPU memory (GB)":42} n/a')
    print(f'{"Current GPU mem during inference (GB)":42} {_f2(gpu_cur)}')
    print(f'{"LiDAR scan frequency (Hz)":42} {_f2(lidar_hz)}')
    print(f'{"Robot average speed (m/s)":42} {_f2(speed)}')
    print(f'{"Avg waypoint execution time (s)":42} {_f2(wp_exec)}')
    print(f'{"Avg waypoint distance (m)":42} {_f2(wp_dist)}')
    if rq_last:
        tot = sum(rq_last) or 1
        p, w, d, ng = rq_last
        print(f'{"Early re-query breakdown":42} '
              f'periodic {100*p/tot:.0f}%, waypoint {100*w/tot:.0f}%, '
              f'detection {100*d/tot:.0f}%, new_goal {100*ng/tot:.0f}%  '
              f'(counts p={p} w={w} d={d} ng={ng})')
    else:
        print(f'{"Early re-query breakdown":42} n/a (no [re_query_stats] yet)')
    print(f'{"Total decision cycles measured":42} {n_cycles}')
    print(f'{"Total episodes":42} {episodes}')
    print('=' * 64)


if __name__ == '__main__':
    main()
