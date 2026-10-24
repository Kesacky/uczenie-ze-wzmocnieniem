#!/usr/bin/env python3
"""Porównanie wariantów z kilku katalogów results/train/ na wszystkich mapach.

Dla każdej pary (algorytm, wariant) i mapy podaje średni zwrot ewaluacji końcowej
(średnia ± odchylenie standardowe po ziarnach), odsetek wyjść, zawałów i liczbę
nieudanych uruchomień (mniej niż 50% bezpiecznych wyjść). Zapisuje tabelę CSV
i wykres słupkowy z heurystyką progową jako punktem odniesienia.

    python scripts/compare_runs.py --runs nagrody_a nagrody_b nagrody_c --out results/train/nagrody
    python scripts/compare_runs.py --runs nagrody_a nagrody_c --select sarsa:q30 sarsa:q30_m1 --out wynik
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
    parser.add_argument("--runs", nargs="+", required=True, help="podkatalogi results/train/")
    parser.add_argument("--out", type=Path, required=True, help="katalog na comparison.csv i comparison.png")
    parser.add_argument("--fail-exit-rate", type=float, default=0.5,
                        help="uruchomienie z mniejszym odsetkiem wyjść liczone jest jako nieudane")
    parser.add_argument("--select", nargs="+", help="tylko te pary algorytm:wariant (w podanej kolejności)")
    parser.add_argument("--png", default="comparison.png", help="nazwa pliku wykresu w katalogu --out")
    args = parser.parse_args()

    # (agent, wariant) -> mapa -> {ziarno: wiersz runs.csv}
    data: dict[tuple, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for name in args.runs:
        for r in read_csv(ROOT / "results" / "train" / name / "runs.csv"):
            variant = r.get("variant") or f"q{float(r['q_init']):g}"   # starsze pliki: tylko q_init
            data[(r["agent"], variant)][r["map"]][int(r["seed"])] = r
    baseline = {r["map"]: float(r["mean_return"]) for r in read_csv(ROOT / "results" / "baseline_summary.csv")
                if r["agent"] == "threshold"}
    if args.select:
        wanted = [tuple(item.split(":", 1)) for item in args.select]
        missing = [w for w in wanted if w not in data]
        if missing:
            parser.error(f"brak wariantów: {missing}")
        data = {w: data[w] for w in wanted}
    maps = sorted({m for per_map in data.values() for m in per_map})

    table = []
    for (agent, variant), per_map in data.items():
        for m in maps:
            runs = list(per_map.get(m, {}).values())
            if not runs:
                continue
            ret = np.array([float(r["mean_return"]) for r in runs])
            table.append({
                "agent": agent, "variant": variant, "map": m, "seeds": len(runs),
                "mean_return": ret.mean(), "sd_return": ret.std(ddof=1) if len(ret) > 1 else 0.0,
                "exit_rate": np.mean([float(r["exit_rate"]) for r in runs]),
                "collapse_rate": np.mean([float(r["collapse_rate"]) for r in runs]),
                "oil_rate": np.mean([float(r["oil_rate"]) for r in runs]),
                "mean_value_out": np.mean([float(r["mean_value_out"]) for r in runs]),
                "failed_runs": sum(float(r["exit_rate"]) < args.fail_exit_rate for r in runs),
                "baseline": baseline.get(m, ""),
            })
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        for row in table:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    print(f"{'algorytm':<12}{'wariant':<28}" + "".join(f"{m[:14]:>22}" for m in maps))
    for agent, variant in data:
        cells = []
        for m in maps:
            row = next((t for t in table if (t["agent"], t["variant"], t["map"]) == (agent, variant, m)), None)
            cells.append(f"{row['mean_return']:7.1f} ±{row['sd_return']:5.1f} ({row['failed_runs']} nieud.)"
                         if row else "-")
        print(f"{agent:<12}{variant:<28}" + "".join(f"{c:>22}" for c in cells))
    print(f"{'heurystyka':<40}" + "".join(f"{baseline.get(m, float('nan')):>22.1f}" for m in maps))

    keys = list(data)
    fig, axes = plt.subplots(1, len(maps), figsize=(4.2 * len(maps), 4.6), squeeze=False, sharey=True)
    colors = plt.get_cmap("tab20").colors if len(keys) > 10 else plt.get_cmap("tab10").colors
    for ax, m in zip(axes.flat, maps):
        for i, key in enumerate(keys):
            row = next((t for t in table if (t["agent"], t["variant"], t["map"]) == (*key, m)), None)
            if row is None:
                continue
            ax.bar(i, row["mean_return"], yerr=row["sd_return"], capsize=2,
                   color=colors[i % len(colors)], hatch="//" if key[0] == "q_learning" else None,
                   edgecolor="white", label=f"{LABELS.get(key[0], key[0])}, {key[1]}")
        if m in baseline:
            ax.axhline(baseline[m], color="black", ls="--", lw=1, label="heurystyka progowa")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(m)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
    axes.flat[0].set_ylabel("średni zwrot ewaluacji [pkt]")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    rows_legend = (len(labels) + 2) // 3
    fig.tight_layout(rect=(0, 0.04 + 0.045 * rows_legend, 1, 1))
    fig.savefig(args.out / args.png, dpi=130)
    print(f"Zapisano {args.out / 'comparison.csv'} i {args.out / args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
