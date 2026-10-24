#!/usr/bin/env python3
"""Trening i ewaluacja Q-learningu i SARSA, wyniki do results/train/<nazwa>/.

Każda kombinacja (mapa, algorytm, q_init, ziarno) to jedno niezależne uruchomienie.
Ziarna są rozłączne między fazami, zgodnie z planem eksperymentów w raporcie:
- trening uruchomienia o ziarnie s: epizody z ziarnami s * 100 000 + i (i < 100 000);
- walidacja w trakcie treningu (co --eval-every epizodów, polityka zachłanna):
  ziarna od 2 000 000, te same dla wszystkich uruchomień;
- ewaluacja końcowa: ziarna od 1 000 000, te same co dla agentów bazowych
  (scripts/run_baselines.py), więc porównanie odbywa się na identycznych przebiegach.

Pliki wynikowe:
- config.json      - argumenty, hiperparametry i konfiguracja gry każdej mapy;
- curves.csv       - przebieg uczenia: średni zwrot treningowy w oknie i wyniki walidacji;
- final_eval.csv   - wszystkie epizody ewaluacji końcowej (format jak baseline_*.csv);
- runs.csv         - podsumowanie każdego uruchomienia;
- summary.csv      - średnia i odchylenie standardowe po ziarnach;
- models/*.pkl     - tablice Q (TabularAgent.save), do podglądu w scripts/watch.py.

    python scripts/train.py --maps m1_stabilna --seeds 3 --episodes 20000 --q-init 0 10 20
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from miner_rl.agents.tabular import (EpsilonSchedule, QLearningAgent, SarsaAgent,  # noqa: E402
                                     evaluate, run_episode)
from miner_rl.env import MineEnv  # noqa: E402

ALGOS = {"q_learning": QLearningAgent, "sarsa": SarsaAgent}
TRAIN_SEED_STRIDE = 100_000
VALID_SEED = 2_000_000
EVAL_FIELDS = ["phase", "map", "agent", "param", "episode", "seed", "return", "length", "reason",
               "collected", "collected_value", "lost_value", "collapsed", "max_depth"]
METRICS = ["mean_return", "std_return", "exit_rate", "collapse_rate", "oil_rate",
           "mean_value_out", "mean_length", "mean_max_depth"]


def make_env(name: str) -> MineEnv:
    return MineEnv(random_maps=True, obs_mode="local") if name == "losowa" else MineEnv(name)


def episode_row(result: dict, episode: int, seed: int) -> dict:
    info = result["info"]
    return {"episode": episode, "seed": seed, "return": round(result["return"], 4),
            "length": result["length"], "reason": info["reason"], "collected": info["collected"],
            "collected_value": info["collected_value"] if info["reason"] == "exit" else 0,
            "lost_value": info["lost_value"], "collapsed": int(info["collapsed"]),
            "max_depth": result["max_depth"]}


def summarize(rows: list[dict]) -> dict:
    """Te same miary co w scripts/run_baselines.py."""
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


def evaluate_rows(env, agent, episodes: int, seed: int) -> list[dict]:
    return [episode_row(r, i, seed + i) for i, r in enumerate(evaluate(env, agent, episodes, seed))]


def run(job: dict) -> dict:
    """Jedno uruchomienie: trening z okresową walidacją, potem ewaluacja końcowa i zapis modelu."""
    env = make_env(job["map"])
    episodes = job["episodes"]
    agent = ALGOS[job["algo"]](
        n_actions=env.action_space.n, alpha=job["alpha"], gamma=job["gamma"], q_init=job["q_init"],
        epsilon=EpsilonSchedule(job["eps_start"], job["eps_end"], int(job["eps_decay"] * episodes)),
        seed=job["seed"])
    train_seed = job["seed"] * TRAIN_SEED_STRIDE
    start = time.perf_counter()
    curve, window = [], []
    for i in range(episodes):
        agent.start_episode(i)
        result = run_episode(env, agent, learn=True, seed=train_seed + i)
        window.append(result)
        if (i + 1) % job["eval_every"] == 0 or i + 1 == episodes:
            valid = summarize(evaluate_rows(env, agent, job["eval_episodes"], VALID_SEED))
            curve.append({
                "episode": i + 1, "epsilon": round(agent.current_epsilon, 4),
                "train_mean_return": round(float(np.mean([w["return"] for w in window])), 4),
                "train_collapse_rate": round(float(np.mean([w["info"]["collapsed"] for w in window])), 4),
                "train_exit_rate": round(float(np.mean([w["info"]["reason"] == "exit" for w in window])), 4),
                **{f"valid_{k}": round(float(v), 4) for k, v in valid.items()},
            })
            window = []
    train_time = time.perf_counter() - start
    final = evaluate_rows(env, agent, job["final_episodes"], job["eval_seed"])
    model = Path(job["out"]) / "models" / f"{job['map']}_{job['algo']}_q{job['q_init']:g}_s{job['seed']}.pkl"
    agent.save(model)
    return {**job, "curve": curve, "final": final, "states": len(agent.q), "train_time": train_time}


def param_label(job: dict) -> str:
    return f"q_init={job['q_init']:g};seed={job['seed']}"


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", nargs="+", default=["m1_stabilna"],
                        help="nazwy map z maps/ albo 'losowa' (nowa kopalnia w każdym epizodzie)")
    parser.add_argument("--algos", nargs="+", default=list(ALGOS), choices=list(ALGOS))
    parser.add_argument("--seeds", type=int, default=3, help="liczba uruchomień (ziarna 0..N-1)")
    parser.add_argument("--episodes", type=int, default=20_000, help="epizody treningu na uruchomienie")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--q-init", type=float, nargs="+", default=[0.0],
                        help="wartości początkowe Q (kilka = osobne warianty)")
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=float, default=0.7, help="ułamek treningu, w którym ε maleje")
    parser.add_argument("--eval-every", type=int, default=500, help="co ile epizodów walidacja")
    parser.add_argument("--eval-episodes", type=int, default=100, help="epizody jednej walidacji")
    parser.add_argument("--final-episodes", type=int, default=1000, help="epizody ewaluacji końcowej")
    parser.add_argument("--eval-seed", type=int, default=1_000_000, help="pierwsze ziarno ewaluacji końcowej")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--name", default="pilot", help="podkatalog w results/train/")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "train")
    args = parser.parse_args()
    if args.episodes > TRAIN_SEED_STRIDE:
        parser.error(f"--episodes może wynosić najwyżej {TRAIN_SEED_STRIDE} (rozłączność ziaren)")

    out = args.out / args.name
    (out / "models").mkdir(parents=True, exist_ok=True)
    jobs = [{"map": m, "algo": a, "q_init": q, "seed": s, "episodes": args.episodes, "alpha": args.alpha,
             "gamma": args.gamma, "eps_start": args.eps_start, "eps_end": args.eps_end,
             "eps_decay": args.eps_decay, "eval_every": args.eval_every, "eval_episodes": args.eval_episodes,
             "final_episodes": args.final_episodes, "eval_seed": args.eval_seed, "out": str(out)}
            for m, a, q, s in product(args.maps, args.algos, args.q_init, range(args.seeds))]
    config = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "train_seed": f"seed * {TRAIN_SEED_STRIDE} + epizod", "valid_seed": VALID_SEED,
              "game_config": {m: make_env(m).config.to_dict() for m in args.maps}}
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(jobs)} uruchomień, {args.workers} procesów -> {out}")
    start = time.perf_counter()
    with Pool(min(args.workers, len(jobs))) as pool:
        results = []
        for r in pool.imap_unordered(run, jobs):
            results.append(r)
            s = summarize(r["final"])
            print(f"  {r['map']} {r['algo']} q_init={r['q_init']:g} ziarno {r['seed']}: "
                  f"zwrot {s['mean_return']:.2f}, wyjścia {s['exit_rate']:.1%}, "
                  f"{r['states']} stanów, {r['train_time']:.0f} s")
    results.sort(key=lambda r: (r["map"], r["algo"], r["q_init"], r["seed"]))

    curves, final_rows, runs = [], [], []
    for r in results:
        key = {"map": r["map"], "agent": r["algo"], "q_init": r["q_init"], "seed": r["seed"]}
        curves += [{**key, **c} for c in r["curve"]]
        for row in r["final"]:
            final_rows.append({"phase": "eval", "map": r["map"], "agent": r["algo"],
                               "param": param_label(r), **row})
        runs.append({**key, "states": r["states"], "train_time_s": round(r["train_time"], 1),
                     **summarize(r["final"])})
    write_csv(out / "curves.csv", curves)
    write_csv(out / "final_eval.csv", final_rows, EVAL_FIELDS)
    write_csv(out / "runs.csv", runs)

    summary = []
    for (m, a, q) in dict.fromkeys((r["map"], r["agent"], r["q_init"]) for r in runs):
        group = [r for r in runs if (r["map"], r["agent"], r["q_init"]) == (m, a, q)]
        row = {"map": m, "agent": a, "q_init": q, "seeds": len(group), "episodes": args.episodes}
        for k in METRICS:
            vals = np.array([g[k] for g in group], dtype=float)
            row[f"{k}_mean"] = vals.mean()
            row[f"{k}_sd"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        summary.append(row)
    write_csv(out / "summary.csv", summary)

    print(f"\nEwaluacja końcowa ({args.final_episodes} epizodów, ziarna od {args.eval_seed}); "
          f"średnia ± odch. std. po {args.seeds} ziarnach, czas {time.perf_counter() - start:.0f} s")
    print(f"{'mapa':<20}{'agent':<12}{'q_init':>7}{'zwrot':>9}{'±':>7}{'wyjście':>9}{'zawał':>8}{'olej':>7}")
    for r in summary:
        print(f"{r['map']:<20}{r['agent']:<12}{r['q_init']:>7g}{r['mean_return_mean']:>9.2f}"
              f"{r['mean_return_sd']:>7.2f}{r['exit_rate_mean']:>9.1%}{r['collapse_rate_mean']:>8.1%}"
              f"{r['oil_rate_mean']:>7.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
