#!/usr/bin/env python3
"""Analiza zachowania wytrenowanej polityki (lub heurystyki) na stałej mapie.

Rozgrywa epizody polityką zachłanną na ziarnach ewaluacji (od 1 000 000) i podaje:
- jak często wydobywane jest każde złoże (z jego rodzajem, wartością i położeniem),
- najczęstsze zestawy wydobytych złóż i przyczyny końca epizodu,
- liczbę akcji niedozwolonych i stawianych stempli na epizod,
- mapę cieplną odwiedzin pól (PNG, opcja --heatmap).

    python scripts/analyze_policy.py --map m1_stabilna --agent results/train/pilot/models/m1_stabilna_sarsa_q10_s0.pkl
    python scripts/analyze_policy.py --map m1_stabilna --agent threshold --threshold 45 --heatmap h.png
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from miner_rl.agents.baselines import ThresholdAgent  # noqa: E402
from miner_rl.agents.loader import make_agent  # noqa: E402
from miner_rl.agents.tabular import act_in_env  # noqa: E402
from miner_rl.env import MineEnv  # noqa: E402

NAMES = {"c": "węgiel", "s": "srebro", "g": "złoto", "k": "kryształ"}


def analyze(env: MineEnv, agent, episodes: int, seed: int) -> dict:
    visits = np.zeros((env.height, env.width), dtype=np.int64)
    sets, reasons = Counter(), Counter()
    mined = np.zeros(env.n_deposits)
    illegal, props, returns = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(agent, "reset"):
            agent.reset()
        visits[env.pos[1], env.pos[0]] += 1
        done, total, n_illegal, n_props = False, 0.0, 0, 0
        while not done:
            obs, reward, term, trunc, info = env.step(act_in_env(agent, env, obs))
            total += reward
            n_illegal += int(info["illegal"])
            n_props += int(info["action"] == "podepnij" and not info["illegal"])
            visits[env.pos[1], env.pos[0]] += 1
            done = term or trunc
        mined += env.mined
        sets[tuple(int(m) for m in env.mined)] += 1
        reasons[info["reason"]] += 1
        illegal.append(n_illegal); props.append(n_props); returns.append(total)
    return {"visits": visits, "sets": sets, "reasons": reasons, "mined_rate": mined / episodes,
            "illegal": np.mean(illegal), "props": np.mean(props), "return": np.mean(returns)}


def describe_set(env: MineEnv, mask: tuple) -> str:
    chosen = [f"{NAMES[ch]} {env.config.deposit_values[ch]}" for (x, y, ch), m in zip(env.deposits, mask) if m]
    return " + ".join(chosen) if chosen else "nic"


def save_heatmap(env: MineEnv, visits: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5.5))
    data = np.where(np.array([[ch == "#" for ch in row] for row in env.grid]), np.nan, visits / visits.sum())
    im = ax.imshow(data, cmap="viridis")
    for x, y, ch in env.deposits:
        ax.text(x, y, ch, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="udział odwiedzin pola")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", required=True)
    parser.add_argument("--agent", required=True, help="plik .pkl, 'threshold' albo 'random'")
    parser.add_argument("--threshold", type=float, default=45.0)
    parser.add_argument("--prop-at", type=int, default=3, help="próg stempla heurystyki")
    parser.add_argument("--oil-bins", type=int, default=4, help="jak przy treningu agenta")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--heatmap", type=Path, help="zapisz mapę cieplną odwiedzin do PNG")
    args = parser.parse_args()

    env = MineEnv(args.map, oil_bins=args.oil_bins)
    if args.agent == "threshold":
        agent = ThresholdAgent.for_env(env, threshold=args.threshold, prop_at=args.prop_at)
    else:
        agent = make_agent(args.agent, env, seed=0, threshold=args.threshold)
    res = analyze(env, agent, args.episodes, args.seed)

    print(f"{args.map}, agent {args.agent}: {args.episodes} epizodów, średni zwrot {res['return']:.2f}")
    print(f"akcje niedozwolone na epizod: {res['illegal']:.2f}, stemple na epizod: {res['props']:.2f}")
    print("przyczyny końca: " + ", ".join(f"{k} {v / args.episodes:.1%}" for k, v in res["reasons"].most_common()))
    print("wydobycie poszczególnych złóż:")
    for (x, y, ch), rate in zip(env.deposits, res["mined_rate"]):
        print(f"  {NAMES[ch]:<9} {env.config.deposit_values[ch]:>3} pkt  pole ({x},{y})  {rate:6.1%}")
    print("najczęstsze zestawy wydobytych złóż (na koniec epizodu):")
    for mask, n in res["sets"].most_common(5):
        print(f"  {n / args.episodes:6.1%}  {describe_set(env, mask)}")
    if args.heatmap:
        save_heatmap(env, res["visits"], args.heatmap, f"{args.map}: {Path(args.agent).stem}")
        print(f"mapa cieplna: {args.heatmap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
