import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supply_chain_env import SupplyChainEnv, ECHELON_NAMES
from agents import (TabularAgent, RLPolicy, discretize, action_to_order,
                    pass_through, order_up_to, smoothed_order_up_to)

OUT = "results"
TEST_SEED0 = 10_000
VAL_SEED0 = 5_000


def shape_reward(local, mode, w=0.5):
    total = local.sum()
    if mode == "local":
        return local
    if mode == "team":
        return np.full_like(local, total)
    return local + w * (total - local)


def train(method, episodes, reward_mode="hybrid", seed=0, log_every=500,
          val_every=100, n_val=40):
    env, val_env = SupplyChainEnv(), SupplyChainEnv()
    agents = [TabularAgent(method, seed=seed + i) for i in range(env.n)]
    curve, val_curve = [], []
    best_val, best_Q = -np.inf, [ag.Q.copy() for ag in agents]
    for ep in range(episodes):
        for ag in agents:
            ag.set_schedule(ep / episodes)
        obs = env.reset(seed=seed * 1_000_000 + ep)
        s = [discretize(o) for o in obs]
        a = [ag.act(si) for ag, si in zip(agents, s)]
        ep_ret, done = 0.0, False
        while not done:
            orders = [action_to_order(o, ai) for o, ai in zip(obs, a)]
            obs, local, done, _ = env.step(orders)
            r = shape_reward(local, reward_mode) / 10.0
            s2 = [discretize(o) for o in obs]
            a2 = [ag.act(si) for ag, si in zip(agents, s2)]
            for i, ag in enumerate(agents):
                ag.update(s[i], a[i], r[i], s2[i], a2[i], done)
            s, a = s2, a2
            ep_ret += local.sum()
        curve.append(ep_ret)
        if (ep + 1) % val_every == 0:
            pol = RLPolicy(agents, method)
            v = np.mean([run_episode(val_env, pol, VAL_SEED0 + k)[0]["profit"]
                         for k in range(n_val)])
            val_curve.append((ep + 1, v))
            if v > best_val:
                best_val, best_Q = v, [ag.Q.copy() for ag in agents]
        if (ep + 1) % log_every == 0:
            print(f"  [{method}/{reward_mode}] bölüm {ep+1:5d}  ε={agents[0].eps:.2f}  "
                  f"son {log_every} ort. kâr = {np.mean(curve[-log_every:]):7.1f}  "
                  f"doğrulama = {val_curve[-1][1]:7.1f}  (en iyi {best_val:7.1f})")
    for ag, Q in zip(agents, best_Q):
        ag.Q = Q
    return agents, np.array(curve), np.array(val_curve)


def run_episode(env, policy, seed):
    obs = env.reset(seed=seed)
    done, tot = False, dict(profit=0.0, hold=0.0, back=0.0, smooth=0.0, dem=0.0, served_now=0.0)
    while not done:
        obs, local, done, info = env.step(policy(obs))
        tot["profit"] += local.sum()
        tot["hold"] += info["hold"].sum()
        tot["back"] += info["back"].sum()
        tot["smooth"] += info["smooth"].sum()
        tot["dem"] += info["customer_demand"]
    H = {k: np.array(v) for k, v in env.hist.items()}
    D = env.demand
    var_d = max(D.var(), 1e-6)
    bw = H["orders"].var(axis=0) / var_d
    new_back = np.maximum(0, np.diff(np.r_[0, H["backlog"][:, 0]]))
    fill = 1 - new_back.sum() / max(D.sum(), 1)
    return dict(profit=tot["profit"], hold=tot["hold"], back=tot["back"], smooth=tot["smooth"],
                fill_rate=fill, bullwhip=bw, avg_inv=H["inv"].mean(axis=0),
                avg_backlog=H["backlog"].mean(axis=0)), H, D


def evaluate(policies, n_test):
    env = SupplyChainEnv()
    res = {}
    for name, pol in policies.items():
        rows = [run_episode(env, pol, TEST_SEED0 + k)[0] for k in range(n_test)]
        res[name] = dict(
            profit=float(np.mean([r["profit"] for r in rows])),
            profit_std=float(np.std([r["profit"] for r in rows])),
            hold=float(np.mean([r["hold"] for r in rows])),
            back=float(np.mean([r["back"] for r in rows])),
            smooth=float(np.mean([r["smooth"] for r in rows])),
            fill_rate=float(np.mean([r["fill_rate"] for r in rows])),
            bullwhip=np.mean([r["bullwhip"] for r in rows], axis=0).tolist(),
            avg_inv=np.mean([r["avg_inv"] for r in rows], axis=0).tolist(),
            avg_backlog=np.mean([r["avg_backlog"] for r in rows], axis=0).tolist(),
        )
    return res


def plots(curves, res, policies, val_curves):
    os.makedirs(OUT, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    for j, (name, c) in enumerate(curves.items()):
        k = 100
        sm = np.convolve(c, np.ones(k) / k, mode="valid")
        plt.plot(sm, color=f"C{j}", alpha=.35, lw=1, label=f"{name} – eğitim")
        vc = val_curves[name]
        plt.plot(vc[:, 0], vc[:, 1], color=f"C{j}", lw=2, label=f"{name} – doğrulama")
    for name in ("Order-up-to (base-stock)", "Yumuşatılmış order-up-to"):
        plt.axhline(res[name]["profit"], ls="--", lw=1, color="gray")
        plt.text(0, res[name]["profit"], " " + name, va="bottom", fontsize=8, color="gray")
    plt.xlabel("Eğitim bölümü"); plt.ylabel("Bölüm başına zincir kârı (100'lük hareketli ort.)")
    plt.title("Öğrenme eğrileri"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"{OUT}/1_ogrenme_egrisi.png", dpi=130); plt.close()

    names = list(res)
    x = np.arange(len(ECHELON_NAMES)); w = 0.8 / len(names)
    plt.figure(figsize=(9, 4.5))
    for j, n in enumerate(names):
        plt.bar(x + j * w, res[n]["bullwhip"], w, label=n)
    plt.axhline(1, color="k", lw=1, ls=":")
    plt.xticks(x + w * (len(names) - 1) / 2, ECHELON_NAMES)
    plt.ylabel("Var(sipariş) / Var(müşteri talebi)"); plt.yscale("log")
    plt.title("Kademe bazında kamçı etkisi oranı (düşük = iyi, 1 = büyütme yok)")
    plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y"); plt.tight_layout()
    plt.savefig(f"{OUT}/2_kamci_orani.png", dpi=130); plt.close()

    env = SupplyChainEnv()
    show = ["Order-up-to (base-stock)", "Q-öğrenme (MARL)"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, n in enumerate(show):
        _, H, D = run_episode(env, policies[n], TEST_SEED0 + 3)
        ax = axes[0, col]
        ax.plot(D, "k-", lw=2, label="Müşteri talebi")
        for i, en in enumerate(ECHELON_NAMES):
            ax.plot(H["orders"][:, i], label=f"{en} siparişi")
        ax.set_title(f"{n}: siparişler"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        ax = axes[1, col]
        for i, en in enumerate(ECHELON_NAMES):
            ax.plot(H["inv"][:, i] - H["backlog"][:, i], label=en)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(f"{n}: net stok (stok − backorder)"); ax.set_xlabel("Hafta")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/3_ornek_bolum.png", dpi=130); plt.close()

    plt.figure(figsize=(9, 4.5))
    comp = [("hold", "Stok tutma"), ("back", "Yok satma cezası"), ("smooth", "Dalgalanma cezası")]
    bottom = np.zeros(len(names))
    for k, lbl in comp:
        v = np.array([res[n][k] for n in names])
        plt.bar(names, v, bottom=bottom, label=lbl); bottom += v
    plt.xticks(rotation=15, fontsize=8); plt.ylabel("Bölüm başına maliyet")
    plt.title("Maliyet kırılımı"); plt.legend(); plt.grid(alpha=.3, axis="y"); plt.tight_layout()
    plt.savefig(f"{OUT}/4_maliyet_kirilimi.png", dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--reward", default="hybrid", choices=["local", "team", "hybrid"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    t0 = time.time()
    curves, trained, val_curves = {}, {}, {}
    for method, label in (("qlearning", "Q-öğrenme (MARL)"), ("sarsa", "SARSA (MARL)")):
        print(f"\n== {label} eğitiliyor ==")
        agents, curve, vcurve = train(method, args.episodes, args.reward)
        curves[label], trained[label] = curve, agents
        val_curves[label] = vcurve
        np.save(f"{OUT}/Q_{method}.npy", np.stack([a.Q for a in agents]))
    print(f"\nEğitim süresi: {time.time() - t0:.1f} sn")

    policies = {
        "Pass-through": pass_through,
        "Order-up-to (base-stock)": order_up_to,
        "Yumuşatılmış order-up-to": smoothed_order_up_to,
        **{k: RLPolicy(v, k) for k, v in trained.items()},
    }
    res = evaluate(policies, args.test)
    with open(f"{OUT}/sonuclar.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"\n== Test sonuçları ({args.test} görülmemiş bölüm) ==")
    hdr = f"{'Politika':28s} {'Kâr':>9s} {'Dolum%':>7s} " + \
          " ".join(f"{'BW-' + e[:4]:>8s}" for e in ECHELON_NAMES) + f" {'Stok':>7s} {'Yoksat':>7s}"
    print(hdr); print("-" * len(hdr))
    for n, r in res.items():
        print(f"{n:28s} {r['profit']:9.1f} {100*r['fill_rate']:7.1f} " +
              " ".join(f"{b:8.2f}" for b in r["bullwhip"]) +
              f" {r['hold']:7.1f} {r['back']:7.1f}")

    plots(curves, res, policies, val_curves)
    print(f"\nGrafikler ve sonuçlar '{OUT}/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
