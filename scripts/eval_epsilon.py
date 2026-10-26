"""Czy krążenie kosztuje punkty: ocena najlepszego agenta z narratorem z małym ε przy grze.

Losowy ruch z prawdopodobieństwem ε wyrywa agenta z pętli. Jeśli wynik rośnie, krążenie
realnie obniża wynik; jeśli nie, dotyczy głównie gier i tak przegranych. Wyniki w raporcie,
sekcja o narratorze.
"""
import csv
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from miner_rl.env.mine_env import MineEnv
from miner_rl.agents.loader import make_agent

RUN, VARIANT, MEMORY = "narrator_pamiec_300k", "q30_n300000_ho3", 1
EPSILONS = [0.0, 0.02, 0.05, 0.1]
EPISODES, SEEDS = 1000, 10


def run(job):
    seed, eps = job
    model = ROOT / f"results/train/{RUN}/models/losowa_q_learning_{VARIANT}_s{seed}.pkl"
    env = MineEnv(random_maps=True, obs_mode="hint", hint_kind="path", hint_memory=MEMORY)
    agent = make_agent(str(model), env)
    agent.rng = np.random.default_rng(seed)
    agent.current_epsilon = eps
    rows = []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=1_000_000 + ep)
        visits, done, total = Counter([env.state_key()]), False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(agent.act(obs, greedy=False, mask=env.action_mask()))
            visits[env.state_key()] += 1
            total, done = total + r, term or trunc
        rows.append({"seed": seed, "epsilon": eps, "return": total, "reason": info["reason"],
                     "circling": max(visits.values()) >= 3})
    return rows


if __name__ == "__main__":
    jobs = [(s, e) for e in EPSILONS for s in range(SEEDS)]
    with ProcessPoolExecutor(8) as pool:
        results = list(pool.map(run, jobs))
    header = ["epsilon", "return_mean", "return_sd", "exit_rate", "collapse_rate", "circling_rate",
              "return_circling", "return_no_circling", "exit_circling", "exit_no_circling"]
    table = []
    for eps in EPSILONS:
        runs = [rows for rows in results if rows[0]["epsilon"] == eps]
        ret = [np.mean([r["return"] for r in rows]) for rows in runs]     # średnia na ziarno
        rate = lambda f: np.mean([np.mean([f(r) for r in rows]) for rows in runs])
        table.append([eps, round(np.mean(ret), 2), round(np.std(ret, ddof=1), 2),
                      round(rate(lambda r: r["reason"] == "exit"), 3),
                      round(rate(lambda r: r["reason"] == "collapse"), 3),
                      round(rate(lambda r: r["circling"]), 3)])
        games = [r for rows in runs for r in rows]                       # gry z krążeniem i bez
        for key in ("return", "exit"):
            for flag in (True, False):
                vals = [r["return"] if key == "return" else r["reason"] == "exit"
                        for r in games if r["circling"] == flag]
                table[-1].append(round(float(np.mean(vals)), 3))
        print(dict(zip(header, table[-1])))
    out = ROOT / f"results/train/{RUN}/epsilon_eval.csv"
    with out.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows([header, *table])
    print(f"Zapisano {out}")
