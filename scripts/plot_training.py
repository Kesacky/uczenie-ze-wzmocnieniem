#!/usr/bin/env python3
"""Krzywe uczenia z results/train/<nazwa>/curves.csv (wynik scripts/train.py).

Dla każdej mapy jeden panel: oś X to epizod treningu, oś Y to średni zwrot walidacji
zachłannej (100 epizodów na stałych ziarnach). Linia to średnia po ziarnach, pasmo to
± odchylenie standardowe. Linia przerywana to heurystyka progowa z
results/baseline_summary.csv (ewaluacja na 1000 epizodach), punkt odniesienia.

    python scripts/plot_training.py --name pilot
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


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="pilot", help="podkatalog w results/train/")
    parser.add_argument("--metric", default="valid_mean_return", help="kolumna curves.csv na osi Y")
    parser.add_argument("--out", type=Path, help="plik PNG (domyślnie results/train/<nazwa>/curves.png)")
    args = parser.parse_args()

    run_dir = ROOT / "results" / "train" / args.name
    rows = read_csv(run_dir / "curves.csv")
    baseline = {r["map"]: float(r["mean_return"]) for r in read_csv(ROOT / "results" / "baseline_summary.csv")
                if r["agent"] == "threshold"}

    # (mapa, agent, q_init) -> epizod -> lista wartości po ziarnach
    series: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        series[(r["map"], r["agent"], float(r["q_init"]))][int(r["episode"])].append(float(r[args.metric]))
    maps = list(dict.fromkeys(r["map"] for r in rows))

    cols = min(2, len(maps))
    nrows = (len(maps) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(6.2 * cols, 4.2 * nrows), squeeze=False, sharex=True)
    for ax, name in zip(axes.flat, maps):
        for (m, agent, q_init), points in series.items():
            if m != name:
                continue
            ep = np.array(sorted(points))
            vals = [points[e] for e in ep]
            mean = np.array([np.mean(v) for v in vals])
            sd = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in vals])
            line, = ax.plot(ep, mean, lw=1.4, label=f"{LABELS.get(agent, agent)}, $Q_0={q_init:g}$",
                            ls="-" if agent == "q_learning" else "-.")
            ax.fill_between(ep, mean - sd, mean + sd, color=line.get_color(), alpha=0.15, lw=0)
        if name in baseline:
            ax.axhline(baseline[name], color="black", ls="--", lw=1, label="heurystyka progowa")
        ax.set_title(name)
        ax.set_xlabel("epizod treningu")
        ax.set_ylabel("średni zwrot walidacji [pkt]")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(maps):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = args.out or run_dir / "curves.png"
    fig.savefig(out, dpi=130)
    print(f"Zapisano {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
