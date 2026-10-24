#!/usr/bin/env python3
"""Wpływ wartości początkowej Q0 na wynik końcowy, zebrany z kilku katalogów results/train/.

Oś X: Q0, oś Y: średni zwrot ewaluacji końcowej (1000 epizodów). Każdy punkt to jedno
uruchomienie (ziarno), linia łączy średnie po ziarnach. Serie rozróżniają liczbę epizodów
treningu i liczbę przedziałów oleju w obserwacji; brane są tylko uruchomienia, w których
pozostałe hiperparametry mają wartości domyślne. Linia przerywana: heurystyka progowa.

    python scripts/plot_q0.py --runs przeglad_m1 q0_m1 olej_m1 --map m1_stabilna
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"q_learning": "Q-learning", "sarsa": "SARSA"}
DEFAULTS = {"alpha": 0.1, "gamma": 0.99, "eps_decay": 0.7, "eps_end": 0.05}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="podkatalogi results/train/ z plikiem runs.csv")
    parser.add_argument("--map", default="m1_stabilna")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "train" / "q0_m1" / "q0_wplyw.png")
    args = parser.parse_args()

    # (agent, seria) -> q0 -> {ziarno: zwrot}; słownik po ziarnie usuwa duplikaty między katalogami
    data: dict[tuple, dict[float, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for name in args.runs:
        with (ROOT / "results" / "train" / name / "runs.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["map"] != args.map or any(float(r.get(k, v)) != v for k, v in DEFAULTS.items()):
                    continue
                bins = int(float(r.get("oil_bins", 4)))
                series = f"{int(float(r['episodes'])) // 1000} tys. epizodów, olej: {bins} przedz."
                data[(r["agent"], series)][float(r["q_init"])][int(r["seed"])] = float(r["mean_return"])
    baseline = None
    with (ROOT / "results" / "baseline_summary.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["map"] == args.map and r["agent"] == "threshold":
                baseline = float(r["mean_return"])

    agents = sorted({a for a, _ in data})
    fig, axes = plt.subplots(1, len(agents), figsize=(6.2 * len(agents), 4.4), squeeze=False, sharey=True)
    for ax, agent in zip(axes.flat, agents):
        for (a, series), points in sorted(data.items()):
            if a != agent:
                continue
            q0 = np.array(sorted(points))
            mean = [np.mean(list(points[q].values())) for q in q0]
            line, = ax.plot(q0, mean, marker="o", lw=1.4, label=series)
            for q in q0:
                ys = list(points[q].values())
                ax.scatter([q] * len(ys), ys, s=14, color=line.get_color(), alpha=0.45, lw=0)
        if baseline is not None:
            ax.axhline(baseline, color="black", ls="--", lw=1, label="heurystyka progowa")
        ax.set_title(f"{args.map}, {LABELS.get(agent, agent)}")
        ax.set_xlabel("wartość początkowa $Q_0$")
        ax.set_ylabel("średni zwrot ewaluacji [pkt]")
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Zapisano {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
