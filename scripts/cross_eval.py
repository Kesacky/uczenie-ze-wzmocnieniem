#!/usr/bin/env python3
"""Ocena wytrenowanych tablic Q na innych mapach niż mapa treningu (eksperyment 3).

Każdy model z results/train/<nazwa>/models/ jest oceniany polityką zachłanną na każdej
mapie docelowej z --targets ('losowa' = nowe kopalnie z generatora). Ziarna ewaluacji
są te same co w ewaluacji końcowej treningu i u agentów bazowych (od 1 000 000), a mapy
losowe wyznacza ziarno epizodu, więc żadna z nich nie wystąpiła w treningu (ziarna
treningu < 1 000 000). Wszystkie mapy mają rozmiar 12×11 i 5 złóż, więc obserwacja
z widokiem otoczenia ma ten sam format na każdej mapie. Modele z obserwacją z podpowiedzią
(wariant z 'ho1' lub 'ho2') są oceniane z tą samą obserwacją. Ocena według zasad gry.

Pliki wynikowe w katalogu --out:
- cross_runs.csv    - wynik każdego modelu na każdej mapie docelowej;
- cross_summary.csv - średnia i odchylenie standardowe po ziarnach.

    python scripts/cross_eval.py --runs losowe_lokalne --targets m1_stabilna m2_krucha m3_gleboka m4_zawal_korytarza losowa --out results/train/mapy
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from multiprocessing import Pool
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from miner_rl.agents.tabular import TabularAgent  # noqa: E402
from train import METRICS, evaluate_rows, make_env, summarize, write_csv  # noqa: E402

MODEL_NAME = re.compile(r"^(?P<map>.+?)_(?P<agent>q_learning|sarsa)_(?P<variant>.+)_s(?P<seed>\d+)\.pkl$")


def evaluate_model(job: dict) -> dict:
    agent = TabularAgent.load(job["model"])
    hint = re.search(r"(?:^|_)ho(\d)", job["variant"])
    env = make_env(job["target"], local_obs=1, hint_obs=int(hint[1]) if hint else 0)
    rows = evaluate_rows(env, agent, job["episodes"], job["seed0"])
    return {k: job[k] for k in ("run", "map", "agent", "variant", "seed", "target")} | summarize(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="podkatalogi results/train/ z modelami")
    parser.add_argument("--targets", nargs="+", required=True, help="mapy docelowe ('losowa' = nowe kopalnie)")
    parser.add_argument("--variants", nargs="*", default=None, help="tylko te warianty (domyślnie wszystkie)")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1_000_000, help="pierwsze ziarno ewaluacji")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    models = []
    for run in args.runs:
        for path in sorted((ROOT / "results" / "train" / run / "models").glob("*.pkl")):
            m = MODEL_NAME.match(path.name)
            if m and (args.variants is None or m["variant"] in args.variants):
                models.append({"run": run, "model": str(path), **m.groupdict(), "seed": int(m["seed"])})
    jobs = [{**m, "target": t, "episodes": args.episodes, "seed0": args.seed}
            for m, t in product(models, args.targets)]
    print(f"{len(models)} modeli × {len(args.targets)} map = {len(jobs)} ocen")
    with Pool(max(1, min(args.workers, len(jobs)))) as pool:
        runs = pool.map(evaluate_model, jobs)
    runs.sort(key=lambda r: (r["run"], r["map"], r["agent"], r["variant"], r["target"], r["seed"]))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "cross_runs.csv", runs)

    summary = []
    for key in dict.fromkeys((r["run"], r["map"], r["agent"], r["variant"], r["target"]) for r in runs):
        group = [r for r in runs if (r["run"], r["map"], r["agent"], r["variant"], r["target"]) == key]
        row = dict(zip(["run", "map", "agent", "variant", "target"], key), seeds=len(group))
        for k in METRICS:
            vals = np.array([g[k] for g in group], dtype=float)
            row[f"{k}_mean"] = vals.mean()
            row[f"{k}_sd"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        row["failed_runs"] = sum(g["exit_rate"] < 0.5 for g in group)
        summary.append(row)
    write_csv(args.out / "cross_summary.csv", summary)

    print(f"{'trening':<20}{'agent':<12}{'wariant':<24}{'ocena na':<20}{'zwrot':>8}{'±':>7}{'wyjście':>9}"
          f"{'pętle':>8}{'nieud.':>7}")
    for r in summary:
        print(f"{r['map']:<20}{r['agent']:<12}{r['variant']:<24}{r['target']:<20}{r['mean_return_mean']:>8.2f}"
              f"{r['mean_return_sd']:>7.2f}{r['exit_rate_mean']:>9.1%}{r['loop_rate_mean']:>8.1%}{r['failed_runs']:>7}")
    print(f"Zapisano {args.out / 'cross_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
