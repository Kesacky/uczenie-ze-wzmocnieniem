#!/usr/bin/env python3
"""Trening i ewaluacja Q-learningu i SARSA, wyniki do results/train/<nazwa>/.

Każdy hiperparametr (--q-init, --alpha, --alpha-end, --gamma, --eps-decay, --eps-end, --episodes)
oraz parametry środowiska (--oil-bins: przedziały oleju w obserwacji; --reward-illegal,
--reward-step, --reward-collapse, --dig-fraction, --dig-refund: nagrody w treningu; --mask-illegal:
maskowanie akcji niedozwolonych; --bonus: premia za eksplorację) przyjmuje jedną lub
kilka wartości. Domyślnie warianty to wszystkie kombinacje
(siatka). Z --ofat pierwsza wartość każdego parametru tworzy wariant bazowy, a każda
pozostała wartość daje wariant różniący się od bazowego tylko tym jednym parametrem
(„jeden czynnik naraz”). Każda kombinacja (mapa, algorytm, wariant, ziarno) to jedno
niezależne uruchomienie.

Zmienione nagrody obowiązują tylko w treningu. Walidacja i ewaluacja końcowa zawsze używają
domyślnych nagród gry (te same zasady co dla agentów bazowych), więc zwroty wszystkich
wariantów są porównywalne. W curves.csv kolumny train_* liczone są z nagród treningowych.

Przy --alpha-end > 0 współczynnik uczenia maleje liniowo od --alpha do --alpha-end w ciągu
całego treningu (0 = stałe α), co stabilizuje tablicę Q pod koniec treningu.

Każde uruchomienie zapamiętuje politykę najlepszą według walidacji (najwyższy średni zwrot;
przy remisie wcześniejszą) spośród walidacji po zakończeniu spadku ε (od epizodu
eps_decay · episodes). Wcześniejsze polityki są niedouczone: w rzadko odwiedzanych stanach
mają wartości początkowe Q i potrafią się zapętlać. Do 25.10.2026 (katalog
results/train/stabilizacja/) wybierano spośród wszystkich walidacji.
Ewaluacja końcowa obejmuje obie polityki: ostatnią (wariant
bez przyrostka) i najlepszą według walidacji (wariant z przyrostkiem _best). Ziarna
walidacji są rozłączne z ziarnami ewaluacji końcowej, więc wybór polityki nie zawyża wyniku.

Ziarna są rozłączne między fazami, zgodnie z planem eksperymentów w raporcie:
- trening uruchomienia o ziarnie s: epizody z ziarnami s * 100 000 + i (i < 100 000);
- walidacja w trakcie treningu (co --eval-every epizodów, polityka zachłanna):
  ziarna od 2 000 000, te same dla wszystkich uruchomień;
- ewaluacja końcowa: ziarna od 1 000 000, te same co dla agentów bazowych
  (scripts/run_baselines.py), więc porównanie odbywa się na identycznych przebiegach.

Pliki wynikowe:
- config.json      - argumenty, lista wariantów i konfiguracja gry każdej mapy;
- curves.csv       - przebieg uczenia: średni zwrot treningowy w oknie i wyniki walidacji;
- final_eval.csv   - wszystkie epizody ewaluacji końcowej (format jak baseline_*.csv);
- runs.csv         - podsumowanie każdego uruchomienia;
- summary.csv      - średnia i odchylenie standardowe po ziarnach dla każdego wariantu;
- models/*.pkl     - tablice Q (TabularAgent.save), do podglądu w scripts/watch.py;
                     *_best_s*.pkl to polityka najlepsza według walidacji.

Wynik każdego ukończonego uruchomienia jest od razu zapisywany w partial/. Ponowne
uruchomienie tego samego polecenia pomija gotowe uruchomienia (wznowienie po przerwaniu).
Po zapisaniu plików wynikowych katalog partial/ jest usuwany.

    python scripts/train.py --maps m1_stabilna --seeds 3 --q-init 0 10 20 --name pilot
    python scripts/train.py --ofat --q-init 10 30 60 --alpha 0.1 0.3 --name przeglad
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from multiprocessing import Pool
import os
from pathlib import Path
import pickle
import shutil
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
               "collected", "collected_value", "lost_value", "collapsed", "max_depth", "repeats"]
METRICS = ["mean_return", "std_return", "exit_rate", "collapse_rate", "oil_rate",
           "mean_value_out", "mean_length", "mean_max_depth", "loop_rate", "mean_repeats"]
# hiperparametr -> (skrót w nazwie wariantu i pliku modelu, wartość domyślna)
# Domyślne od 24.10.2026: Q0 = 30, nagrody gry z config.py (kara -2 / -10, premia 0,3) i maskowanie
# akcji niedozwolonych. Od 25.10.2026 także α malejące do 0,01 i kara za krok -0,25 w treningu
# (ocena nadal według zasad gry, -0,1 za krok). Wcześniejsze katalogi results/train/ mają zapisane
# parametry w config.json.
PARAMS = {"q_init": ("q", 30.0), "alpha": ("a", 0.1), "alpha_end": ("ae", 0.01), "gamma": ("g", 0.99),
          "eps_decay": ("d", 0.7), "eps_end": ("e", 0.05), "episodes": ("n", 50_000),
          "oil_bins": ("o", 4), "reward_step": ("rs", -0.25), "reward_illegal": ("ri", -2.0),
          "reward_collapse": ("rc", -10.0),
          "dig_fraction": ("df", 0.3), "dig_refund": ("dr", 0), "mask_illegal": ("m", 1),
          "bonus": ("b", 0.0)}


def make_env(name: str, oil_bins: int = PARAMS["oil_bins"][1], **rewards) -> MineEnv:
    if name == "losowa":
        return MineEnv(random_maps=True, obs_mode="local", oil_bins=oil_bins, **rewards)
    return MineEnv(name, oil_bins=oil_bins, **rewards)


def train_env(job: dict) -> MineEnv:
    """Środowisko treningowe: nagrody wariantu zamiast domyślnych."""
    return make_env(job["map"], job["oil_bins"], reward_step=job["reward_step"],
                    reward_illegal=job["reward_illegal"],
                    reward_collapse=job["reward_collapse"], reward_dig_fraction=job["dig_fraction"],
                    reward_dig_refund=bool(job["dig_refund"]), reward_shaping_gamma=job["gamma"])


def variant_label(v: dict) -> str:
    """Krótka nazwa wariantu: q_init zawsze, pozostałe parametry tylko gdy różne od domyślnych."""
    parts = [f"q{v['q_init']:g}"]
    parts += [f"{short}{v[k]:g}" for k, (short, default) in PARAMS.items()
              if k != "q_init" and v[k] != default]
    return "_".join(parts)


def make_variants(values: dict[str, list], ofat: bool) -> list[dict]:
    if not ofat:
        return [dict(zip(values, combo)) for combo in product(*values.values())]
    base = {k: v[0] for k, v in values.items()}
    variants = [base]
    for k, vals in values.items():
        variants += [{**base, k: x} for x in vals[1:]]
    return variants


def episode_row(result: dict, episode: int, seed: int) -> dict:
    info = result["info"]
    return {"episode": episode, "seed": seed, "return": round(result["return"], 4),
            "length": result["length"], "reason": info["reason"], "collected": info["collected"],
            "collected_value": info["collected_value"] if info["reason"] == "exit" else 0,
            "lost_value": info["lost_value"], "collapsed": int(info["collapsed"]),
            "max_depth": result["max_depth"], "repeats": result["repeats"]}


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
        "loop_rate": sum(r["repeats"] > 0 for r in rows) / n,
        "mean_repeats": np.mean([r["repeats"] for r in rows]),
    }


def evaluate_rows(env, agent, episodes: int, seed: int) -> list[dict]:
    return [episode_row(r, i, seed + i) for i, r in enumerate(evaluate(env, agent, episodes, seed))]


def run(job: dict) -> dict:
    """Jedno uruchomienie: trening z okresową walidacją, potem ewaluacja końcowa i zapis modelu."""
    env = train_env(job)
    eval_env = make_env(job["map"], job["oil_bins"])    # domyślne nagrody gry
    episodes = job["episodes"]
    agent = ALGOS[job["algo"]](
        n_actions=env.action_space.n, alpha=job["alpha"], gamma=job["gamma"], q_init=job["q_init"],
        mask_illegal=bool(job["mask_illegal"]), bonus_beta=job["bonus"],
        epsilon=EpsilonSchedule(job["eps_start"], job["eps_end"], int(job["eps_decay"] * episodes)),
        alpha_schedule=EpsilonSchedule(job["alpha"], job["alpha_end"], episodes) if job["alpha_end"] > 0 else None,
        seed=job["seed"])
    train_seed = job["seed"] * TRAIN_SEED_STRIDE
    start = time.perf_counter()
    curve, window = [], []
    best = {"return": -np.inf, "episode": 0, "q": None}
    best_from = int(job["eps_decay"] * episodes)          # wybór najlepszej polityki po spadku ε
    for i in range(episodes):
        agent.start_episode(i)
        result = run_episode(env, agent, learn=True, seed=train_seed + i)
        window.append(result)
        if (i + 1) % job["eval_every"] == 0 or i + 1 == episodes:
            valid = summarize(evaluate_rows(eval_env, agent, job["eval_episodes"], VALID_SEED))
            curve.append({
                "episode": i + 1, "epsilon": round(agent.current_epsilon, 4),
                "train_mean_return": round(float(np.mean([w["return"] for w in window])), 4),
                "train_collapse_rate": round(float(np.mean([w["info"]["collapsed"] for w in window])), 4),
                "train_exit_rate": round(float(np.mean([w["info"]["reason"] == "exit" for w in window])), 4),
                **{f"valid_{k}": round(float(v), 4) for k, v in valid.items()},
            })
            window = []
            if (i + 1 >= best_from or i + 1 == episodes) and valid["mean_return"] > best["return"]:
                best = {"return": valid["mean_return"], "episode": i + 1, "q": agent.copy_q()}
    train_time = time.perf_counter() - start
    final = evaluate_rows(eval_env, agent, job["final_episodes"], job["eval_seed"])
    models = Path(job["out"]) / "models"
    agent.save(models / f"{job['map']}_{job['algo']}_{job['variant']}_s{job['seed']}.pkl")
    states = len(agent.q)
    agent.q = best["q"]
    final_best = evaluate_rows(eval_env, agent, job["final_episodes"], job["eval_seed"])
    agent.save(models / f"{job['map']}_{job['algo']}_{job['variant']}_best_s{job['seed']}.pkl")
    return {**job, "curve": curve, "final": final, "final_best": final_best, "best_episode": best["episode"],
            "states": states, "train_time": train_time}


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
    parser.add_argument("--q-init", type=float, nargs="+", default=[PARAMS["q_init"][1]],
                        help="wartości początkowe Q")
    parser.add_argument("--alpha", type=float, nargs="+", default=[PARAMS["alpha"][1]])
    parser.add_argument("--alpha-end", type=float, nargs="+", default=[PARAMS["alpha_end"][1]],
                        help="końcowe α przy liniowym spadku w ciągu treningu (0 = stałe α)")
    parser.add_argument("--gamma", type=float, nargs="+", default=[PARAMS["gamma"][1]])
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, nargs="+", default=[PARAMS["eps_end"][1]])
    parser.add_argument("--eps-decay", type=float, nargs="+", default=[PARAMS["eps_decay"][1]],
                        help="ułamek treningu, w którym ε maleje")
    parser.add_argument("--episodes", type=int, nargs="+", default=[PARAMS["episodes"][1]],
                        help="epizody treningu na uruchomienie")
    parser.add_argument("--oil-bins", type=int, nargs="+", default=[PARAMS["oil_bins"][1]],
                        help="przedziały oleju w obserwacji (100 = dokładny poziom oleju)")
    parser.add_argument("--reward-step", type=float, nargs="+", default=[PARAMS["reward_step"][1]],
                        help="kara za każdy krok w treningu")
    parser.add_argument("--reward-illegal", type=float, nargs="+", default=[PARAMS["reward_illegal"][1]],
                        help="kara za akcję niedozwoloną w treningu")
    parser.add_argument("--reward-collapse", type=float, nargs="+", default=[PARAMS["reward_collapse"][1]],
                        help="kara za zawał w treningu")
    parser.add_argument("--dig-fraction", type=float, nargs="+", default=[PARAMS["dig_fraction"][1]],
                        help="ułamek wartości złoża wypłacany przy wydobyciu w treningu")
    parser.add_argument("--dig-refund", type=int, nargs="+", default=[PARAMS["dig_refund"][1]], choices=[0, 1],
                        help="1 = zwrot wypłaty za wydobycie przy końcu epizodu (kształtowanie potencjałowe)")
    parser.add_argument("--mask-illegal", type=int, nargs="+", default=[PARAMS["mask_illegal"][1]], choices=[0, 1],
                        help="1 = agent wybiera tylko akcje dozwolone")
    parser.add_argument("--bonus", type=float, nargs="+", default=[PARAMS["bonus"][1]],
                        help="premia za eksplorację β/√N(s,a) w treningu (0 = wyłączona)")
    parser.add_argument("--ofat", action="store_true",
                        help="warianty „jeden czynnik naraz” wokół pierwszych wartości zamiast siatki")
    parser.add_argument("--eval-every", type=int, default=500, help="co ile epizodów walidacja")
    parser.add_argument("--eval-episodes", type=int, default=100, help="epizody jednej walidacji")
    parser.add_argument("--final-episodes", type=int, default=1000, help="epizody ewaluacji końcowej")
    parser.add_argument("--eval-seed", type=int, default=1_000_000, help="pierwsze ziarno ewaluacji końcowej")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--name", default="pilot", help="podkatalog w results/train/")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "train")
    args = parser.parse_args()
    if max(args.episodes) > TRAIN_SEED_STRIDE:
        parser.error(f"--episodes może wynosić najwyżej {TRAIN_SEED_STRIDE} (rozłączność ziaren)")

    values = {k: getattr(args, k) for k in PARAMS}
    variants = make_variants(values, args.ofat)
    for v in variants:
        v["variant"] = variant_label(v)
    out = args.out / args.name
    (out / "models").mkdir(parents=True, exist_ok=True)
    common = {"eps_start": args.eps_start, "eval_every": args.eval_every, "eval_episodes": args.eval_episodes,
              "final_episodes": args.final_episodes, "eval_seed": args.eval_seed, "out": str(out)}
    jobs = [{"map": m, "algo": a, "seed": s, **v, **common}
            for m, a, v, s in product(args.maps, args.algos, variants, range(args.seeds))]
    # najdłuższe uruchomienia najpierw, żeby procesy kończyły się mniej więcej razem
    jobs.sort(key=lambda j: -j["episodes"])
    config = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "variants": variants, "train_seed": f"seed * {TRAIN_SEED_STRIDE} + epizod",
              "valid_seed": VALID_SEED, "game_config": {m: make_env(m).config.to_dict() for m in args.maps}}
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    partial = out / "partial"
    partial.mkdir(exist_ok=True)
    key = lambda j: f"{j['map']}_{j['algo']}_{j['variant']}_s{j['seed']}"  # noqa: E731
    results = []
    for j in jobs:
        saved = partial / f"{key(j)}.pkl"
        if saved.exists():
            with saved.open("rb") as f:
                results.append(pickle.load(f))
    todo = [j for j in jobs if not (partial / f"{key(j)}.pkl").exists()]
    print(f"{len(variants)} wariantów, {len(jobs)} uruchomień (gotowe wcześniej: {len(results)}), "
          f"{args.workers} procesów -> {out}")
    start = time.perf_counter()
    with Pool(max(1, min(args.workers, len(todo)))) as pool:
        for r in pool.imap_unordered(run, todo):
            results.append(r)
            with (partial / f"{key(r)}.pkl").open("wb") as f:
                pickle.dump(r, f)
            s = summarize(r["final"])
            print(f"  [{len(results)}/{len(jobs)}] {r['map']} {r['algo']} {r['variant']} ziarno {r['seed']}: "
                  f"zwrot {s['mean_return']:.2f}, wyjścia {s['exit_rate']:.1%}, "
                  f"{r['states']} stanów, {r['train_time']:.0f} s", flush=True)
    order = {v["variant"]: i for i, v in enumerate(variants)}
    results.sort(key=lambda r: (r["map"], r["algo"], order[r["variant"]], r["seed"]))

    curves, final_rows, runs = [], [], []
    for r in results:
        key = {"map": r["map"], "agent": r["algo"], "variant": r["variant"],
               **{k: r[k] for k in PARAMS}, "seed": r["seed"]}
        curves += [{**key, **c} for c in r["curve"]]
        # polityka ostatnia i najlepsza według walidacji jako osobne warianty
        for suffix, final in [("", r["final"]), ("_best", r["final_best"])]:
            variant = r["variant"] + suffix
            for row in final:
                final_rows.append({"phase": "eval", "map": r["map"], "agent": r["algo"],
                                   "param": f"{variant};seed={r['seed']}", **row})
            runs.append({**key, "variant": variant,
                         "policy_episode": r["best_episode"] if suffix else r["episodes"],
                         "states": r["states"], "train_time_s": round(r["train_time"], 1), **summarize(final)})
    write_csv(out / "curves.csv", curves)
    write_csv(out / "final_eval.csv", final_rows, EVAL_FIELDS)
    write_csv(out / "runs.csv", runs)

    summary = []
    for (m, a, v) in dict.fromkeys((r["map"], r["agent"], r["variant"]) for r in runs):
        group = [r for r in runs if (r["map"], r["agent"], r["variant"]) == (m, a, v)]
        row = {"map": m, "agent": a, "variant": v, **{k: group[0][k] for k in PARAMS}, "seeds": len(group)}
        for k in METRICS:
            vals = np.array([g[k] for g in group], dtype=float)
            row[f"{k}_mean"] = vals.mean()
            row[f"{k}_sd"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        summary.append(row)
    write_csv(out / "summary.csv", summary)
    shutil.rmtree(partial)

    print(f"\nEwaluacja końcowa ({args.final_episodes} epizodów, ziarna od {args.eval_seed}); "
          f"średnia ± odch. std. po {args.seeds} ziarnach, czas {time.perf_counter() - start:.0f} s")
    print(f"{'mapa':<20}{'agent':<12}{'wariant':<26}{'zwrot':>9}{'±':>7}{'wyjście':>9}{'zawał':>8}"
          f"{'olej':>7}{'wartość':>9}{'pętle':>8}")
    for r in summary:
        print(f"{r['map']:<20}{r['agent']:<12}{r['variant']:<26}{r['mean_return_mean']:>9.2f}"
              f"{r['mean_return_sd']:>7.2f}{r['exit_rate_mean']:>9.1%}{r['collapse_rate_mean']:>8.1%}"
              f"{r['oil_rate_mean']:>7.1%}{r['mean_value_out_mean']:>9.2f}{r['loop_rate_mean']:>8.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
