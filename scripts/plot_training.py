#!/usr/bin/env python3
"""Krzywe uczenia z results/train/<nazwa>/curves.csv (wynik scripts/train.py).

Dla każdej mapy (z --split-agents: dla każdej pary mapa-algorytm) jeden panel: oś X
to epizod treningu, oś Y to średni zwrot walidacji zachłannej (100 epizodów na stałych
ziarnach). Linia to średnia po ziarnach, pasmo to
± odchylenie standardowe. Linia przerywana to heurystyka progowa z
results/baseline_summary.csv (ewaluacja na 1000 epizodach), punkt odniesienia.

    python scripts/plot_training.py --name pilot
    python scripts/plot_training.py --name przeglad_m1 --split-agents --variants q10 q30 q60
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
    parser.add_argument("--variants", nargs="+", help="tylko te warianty (kolumna variant w curves.csv)")
    parser.add_argument("--split-agents", action="store_true", help="osobny panel dla każdego algorytmu")
    parser.add_argument("--out", type=Path, help="plik PNG (domyślnie results/train/<nazwa>/curves.png)")
    args = parser.parse_args()

    run_dir = ROOT / "results" / "train" / args.name
    rows = read_csv(run_dir / "curves.csv")
    baseline = {r["map"]: float(r["mean_return"]) for r in read_csv(ROOT / "results" / "baseline_summary.csv")
                if r["agent"] == "threshold"}

    # (panel, agent, wariant) -> epizod -> lista wartości po ziarnach
    series: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        variant = r.get("variant") or f"q{float(r['q_init']):g}"   # starsze pliki: tylko q_init
        if args.variants and variant not in args.variants:
            continue
        panel = (r["map"], r["agent"]) if args.split_agents else (r["map"], None)
        series[(panel, r["agent"], variant)][int(r["episode"])].append(float(r[args.metric]))
    maps = list(dict.fromkeys(key[0] for key in series))

    cols = min(2, len(maps))
    nrows = (len(maps) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(6.2 * cols, 4.2 * nrows), squeeze=False, sharex=True)
    for ax, panel in zip(axes.flat, maps):
        name = panel[0]
        for (p, agent, variant), points in series.items():
            if p != panel:
                continue
            ep = np.array(sorted(points))
            vals = [points[e] for e in ep]
            mean = np.array([np.mean(v) for v in vals])
            sd = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in vals])
            label = variant if args.split_agents else f"{LABELS.get(agent, agent)}, {variant}"
            line, = ax.plot(ep, mean, lw=1.4, label=label, ls="-" if agent == "q_learning" else "-.")
            ax.fill_between(ep, mean - sd, mean + sd, color=line.get_color(), alpha=0.15, lw=0)
        if name in baseline:
            ax.axhline(baseline[name], color="black", ls="--", lw=1, label="heurystyka progowa")
        ax.set_title(name if panel[1] is None else f"{name}, {LABELS.get(panel[1], panel[1])}")
        ax.set_xlabel("epizod treningu")
        ax.set_ylabel("średni zwrot walidacji [pkt]")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(maps):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    ncol = min(2 * cols, len(labels))   # przy jednym panelu legenda nie mieści się w 4 kolumnach
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False)
    legend_rows = (len(labels) + ncol - 1) // ncol
    fig.tight_layout(rect=(0, (0.05 + 0.035 * legend_rows) / nrows ** 0.5, 1, 1))
    out = args.out or run_dir / "curves.png"
    fig.savefig(out, dpi=130)
    print(f"Zapisano {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
