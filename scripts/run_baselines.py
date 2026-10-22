#!/usr/bin/env python3
"""Ewaluacja agentów bazowych, wyniki do results/baseline_<mapa>.csv.

Mapy: wszystkie stałe z maps/ oraz wariant „losowa” (MineEnv(random_maps=True): w każdym
epizodzie nowa kopalnia z generatora, wyznaczona przez ziarno epizodu).
Procedura zgodna z planem eksperymentów w raporcie:
1. dobór wariantu heurystyki (próg X i moment stawiania stempla) na epizodach
   „treningowych” (ziarna --tune-seed + i), osobno dla każdej mapy, kryterium: średni zwrot;
2. ewaluacja końcowa wybranego wariantu i agenta losowego na rozłącznych ziarnach
   (--seed + i, domyślnie 1 000 000 + i, jak `evaluate` w miner_rl/agents/tabular.py).
Każdy wiersz CSV to jeden epizod (faza, agent, parametr, ziarno, zwrot, długość,
przyczyna końca, wyniesiona i utracona wartość, głębokość). Wszyscy agenci grają przy
tych samych ziarnach środowiska. Podsumowanie: results/baseline_summary.csv.

    python scripts/run_baselines.py --episodes 1000
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from miner_rl.agents import RandomAgent, ThresholdAgent  # noqa: E402
from miner_rl.env import MineEnv, list_maps  # noqa: E402

FIELDS = ["phase", "map", "agent", "param", "episode", "seed", "return", "length", "reason",
          "collected", "collected_value", "lost_value", "collapsed", "max_depth"]
THRESHOLDS = [3, 9, 18, 30, 45]     # progi ładunku X (wartości złóż: 3, 9, 18, 30)
PROP_AT = [1, 3]                     # stempel, gdy stabilność spadnie do tej wartości


def evaluate(env, agent, episodes, seed):
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        agent.reset()
        total, done, max_depth = 0.0, False, 0
        while not done:
            obs, reward, term, trunc, info = env.step(agent.act(obs, greedy=True))
            total += reward
            max_depth = max(max_depth, info["depth"])
            done = term or trunc
        rows.append({
            "episode": ep, "seed": seed + ep, "return": round(total, 4),
            "length": info["steps"], "reason": info["reason"],
            "collected": info["collected"],
            "collected_value": info["collected_value"] if info["reason"] == "exit" else 0,
            "lost_value": info["lost_value"], "collapsed": int(info["collapsed"]),
            "max_depth": max_depth,
        })
    return rows


def summarize(rows):
    returns = np.array([r["return"] for r in rows], dtype=float)
    n = len(rows)
    return {
        "mean_return": returns.mean(), "std_return": returns.std(ddof=1) if n > 1 else 0.0,
        "exit_rate": sum(r["reason"] == "exit" for r in rows) / n,
        "collapse_rate": sum(r["collapsed"] for r in rows) / n,
        "oil_rate": sum(r["reason"] == "out_of_oil" for r in rows) / n,
        "mean_value_out": np.mean([r["collected_value"] for r in rows]),
        "mean_length": np.mean([r["length"] for r in rows]),
        "mean_max_depth": np.mean([r["max_depth"] for r in rows]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=1000, help="epizody ewaluacji")
    parser.add_argument("--tune-episodes", type=int, default=500, help="epizody doboru wariantu na wariant")
    parser.add_argument("--seed", type=int, default=1_000_000, help="pierwsze ziarno ewaluacji")
    parser.add_argument("--tune-seed", type=int, default=0, help="pierwsze ziarno doboru progu")
    parser.add_argument("--maps", nargs="*",
                        help="nazwy map (domyślnie wszystkie z maps/ oraz 'losowa' = nowa mapa z generatora w każdym epizodzie)")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    names = args.maps or [p.stem for p in list_maps()] + ["losowa"]
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in names:
        env = MineEnv(random_maps=True) if name == "losowa" else MineEnv(name)
        start = time.perf_counter()
        rows = []
        # 1. dobór wariantu heurystyki (próg X, moment stawiania stempla) na ziarnach treningowych
        tuning = {}
        for t in THRESHOLDS:
            for pa in PROP_AT:
                param = f"X={t:g};stempel<={pa}"
                agent = ThresholdAgent.for_env(env, threshold=t, prop_at=pa)
                agent_rows = evaluate(env, agent, args.tune_episodes, args.tune_seed)
                for r in agent_rows:
                    r.update(phase="tune", map=name, agent="threshold", param=param)
                rows += agent_rows
                tuning[(t, pa)] = summarize(agent_rows)["mean_return"]
        best_t, best_pa = max(tuning, key=lambda k: tuning[k])
        best = f"X={best_t:g};stempel<={best_pa}"
        # 2. ewaluacja na rozłącznych ziarnach: agent losowy i wybrany wariant heurystyki
        agents = [("random", "-", RandomAgent(n_actions=env.action_space.n, seed=args.seed)),
                  ("threshold", best, ThresholdAgent.for_env(env, threshold=best_t, prop_at=best_pa))]
        for agent_name, param, agent in agents:
            agent_rows = evaluate(env, agent, args.episodes, args.seed)
            for r in agent_rows:
                r.update(phase="eval", map=name, agent=agent_name, param=param)
            rows += agent_rows
            summary.append({"map": name, "agent": agent_name, "param": param,
                            "tune_mean_return": round(tuning[(best_t, best_pa)], 4) if agent_name == "threshold" else "",
                            "episodes": args.episodes, **summarize(agent_rows)})
        path = args.out / f"baseline_{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}: {len(rows)} epizodów w {time.perf_counter() - start:.1f} s, "
              f"wybrany wariant {best} -> {path.relative_to(ROOT)}")

    summary_path = args.out / "baseline_summary.csv"
    keys = list(summary[0])
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in summary:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    print(f"\nEwaluacja ({args.episodes} epizodów, ziarna od {args.seed})")
    print(f"{'mapa':<20}{'agent':<11}{'wariant':<18}{'zwrot':>9}{'±':>7}{'wyjście':>9}{'zawał':>8}{'olej':>7}{'wartość':>9}")
    for r in summary:
        print(f"{r['map']:<20}{r['agent']:<11}{r['param']:<18}{r['mean_return']:>9.2f}{r['std_return']:>7.2f}"
              f"{r['exit_rate']:>9.1%}{r['collapse_rate']:>8.1%}{r['oil_rate']:>7.1%}{r['mean_value_out']:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
