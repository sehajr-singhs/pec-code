"""Seed farm: run experiments across seeds in parallel CPU workers.

Each worker runs one seed of one experiment with the given profile and writes
results/seedfarm/<exp>/<seed>.json. Aggregation then pools per-seed numbers
with the paired-stats machinery (which only needs per-seed values).

Usage:
    python seed_farm.py e3 --seeds 0-9 --workers 8 --profile full
    python seed_farm.py e0,e6 --seeds 0-9 --workers 8 --profile farm
    python seed_farm.py --status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

WORKER = r'''
import json, os, sys, time
import torch
torch.set_num_threads(int(sys.argv[5]))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pec2 import experiments as E

exp, seed, profile, out_path, nthreads = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
from pec2.run_all import PROFILES
p = dict(PROFILES[profile].get(exp, {}))
t0 = time.time()
if exp == "e0":
    r = E.exp0_estimator(seeds=[seed], device="cpu", **p)
elif exp == "e1":
    r = E.exp1_gate_vs_noise(seeds=[seed], device="cpu", log=None, **p)
elif exp == "e3":
    r = E.exp3_fault_adaptation(seeds=[seed], device="cpu", log=None, **p)
elif exp == "e5":
    r = E.exp5_hold_task(seeds=[seed], device="cpu", log=None, **p)
elif exp == "e6":
    r = E.exp6_identifiability(seeds=[seed], device="cpu", log=None, **p)
else:
    raise SystemExit(f"unknown exp {exp}")
r["wall_seconds"] = round(time.time() - t0, 1)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(r, f, indent=1)
print(f"worker done {exp} seed {seed} in {r['wall_seconds']}s")
'''


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="?", default=None,
                    help="comma list: e0,e3,e5,e6")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--profile", default="full",
                    help="run_all profile used per seed (default full)")
    ap.add_argument("--threads", type=int, default=2,
                    help="torch threads per worker")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "seedfarm"))
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        if not os.path.isdir(args.out):
            print("no farm output yet")
            return
        for exp in sorted(os.listdir(args.out)):
            d = os.path.join(args.out, exp)
            if os.path.isdir(d):
                files = [f for f in os.listdir(d) if f.endswith(".json")]
                print(f"{exp}: {len(files)} seeds done")
        return

    if not args.exps:
        ap.error("exps required unless --status")
    seeds = parse_seeds(args.seeds)
    exps = args.exps.split(",")
    jobs = [(e, s) for e in exps for s in seeds]
    print(f"farm: {len(jobs)} jobs ({exps} x {len(seeds)} seeds), "
          f"{args.workers} workers x {args.threads} threads, profile={args.profile}")

    worker_path = os.path.join(HERE, "_farm_worker.py")
    with open(worker_path, "w") as f:
        f.write(WORKER)

    running = []  # (proc, exp, seed, logfile)
    done = 0
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < args.workers:
            e, s = jobs.pop(0)
            out_path = os.path.join(args.out, e, f"seed_{s}.json")
            if os.path.exists(out_path):
                done += 1
                print(f"[{done}/{len(jobs) + done}] {e} seed {s}: cached")
                continue
            log = open(os.path.join(args.out, f"{e}_s{s}.log"), "w")
            proc = subprocess.Popen(
                [sys.executable, worker_path, e, str(s), args.profile,
                 out_path, str(args.threads)],
                cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
            running.append((proc, e, s, log))
        time.sleep(5)
        still = []
        for proc, e, s, log in running:
            if proc.poll() is None:
                still.append((proc, e, s, log))
            else:
                log.close()
                done += 1
                status = "ok" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
                print(f"[{done}/{len(jobs) + done}] {e} seed {s}: {status} "
                      f"({time.time() - t0:.0f}s elapsed)")
        running = still
    print(f"farm complete in {(time.time() - t0) / 60:.1f} min -> {args.out}/")


if __name__ == "__main__":
    main()
