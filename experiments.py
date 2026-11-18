import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from main import train, evaluate, OUT
from agents import RLPolicy, pass_through, order_up_to, smoothed_order_up_to


def job(args):
    method, reward_mode, seed, episodes, n_test = args
    agents, _, _ = train(method, episodes, reward_mode, seed=seed, log_every=10**9)
    r = evaluate({"rl": RLPolicy(agents, method)}, n_test)["rl"]
    return dict(method=method, reward=reward_mode, seed=seed, **r)


def ci95(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    jobs = [(m, "hybrid", s, a.episodes, a.test) for m in ("qlearning", "sarsa") for s in range(a.seeds)]
    jobs += [(m, rm, s, a.episodes, a.test) for m in ("qlearning", "sarsa")
             for rm in ("local", "team") for s in range(3)]
    with Pool(a.workers) as p:
        runs = p.map(job, jobs)

    base = evaluate({"Pass-through": pass_through, "Order-up-to (base-stock)": order_up_to,
                     "Yumuşatılmış order-up-to": smoothed_order_up_to}, a.test)

    summary = {}
    for m in ("qlearning", "sarsa"):
        for rm in ("hybrid", "local", "team"):
            rs = [r for r in runs if r["method"] == m and r["reward"] == rm]
            if not rs:
                continue
            summary[f"{m}/{rm}"] = dict(
                n_seeds=len(rs),
                profit=ci95([r["profit"] for r in rs]),
                fill_rate=ci95([r["fill_rate"] for r in rs]),
                bullwhip=[ci95([r["bullwhip"][i] for r in rs]) for i in range(3)],
                hold=ci95([r["hold"] for r in rs]),
                back=ci95([r["back"] for r in rs]),
                smooth=ci95([r["smooth"] for r in rs]),
            )
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/deneyler.json", "w", encoding="utf-8") as f:
        json.dump(dict(summary=summary, baselines=base, runs=runs), f, ensure_ascii=False, indent=2)

    print(f"{'Yapılandırma':22s} {'n':>2s} {'Kâr':>15s} {'Dolum%':>12s} {'BW-Depo':>12s} {'BW-Fab':>12s} {'BW-Ted':>12s}")
    for k, s in summary.items():
        bw = " ".join(f"{m:5.2f}±{c:4.2f}" for m, c in s["bullwhip"])
        print(f"{k:22s} {s['n_seeds']:2d} {s['profit'][0]:7.0f}±{s['profit'][1]:5.0f} "
              f"{100*s['fill_rate'][0]:6.1f}±{100*s['fill_rate'][1]:3.1f}  {bw}")
    for k, r in base.items():
        print(f"{k:28s} kâr={r['profit']:.0f} dolum={100*r['fill_rate']:.1f} bw={np.round(r['bullwhip'], 2)}")


if __name__ == "__main__":
    main()
