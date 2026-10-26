"""Jak często agent z narratorem krąży (stan gry >= 3 razy w epizodzie); wyniki w raporcie, sekcja o narratorze."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from miner_rl.env.mine_env import MineEnv
from miner_rl.agents.loader import make_agent
from miner_rl.agents.tabular import act_in_env

N = 300
# (katalog wyników, wariant modelu, hint_memory)
RUNS = [("narrator_losowe", "q30_n100000_ho2", 0), ("narrator_pamiec", "q30_n100000_ho3", 1),
        ("narrator_pamiec_300k", "q30_n300000_ho3", 1), ("narrator_licznik", "q30_n100000_ho4", 2)]


def model_path(run, variant, seed):
    return ROOT / f"results/train/{run}/models/losowa_q_learning_{variant}_s{seed}.pkl"


for run, variant, memory in RUNS:
    stuck, maxv, steps_stuck = [], [], []
    for seed in range(10):
        env = MineEnv(random_maps=True, obs_mode="hint", hint_kind="path", hint_memory=memory)
        agent = make_agent(str(model_path(run, variant, seed)), env, 0, 6)
        for ep in range(N):
            obs, _ = env.reset(seed=1_000_000 + ep)
            visits, done = Counter([env.state_key()]), False
            while not done:
                obs, _, term, trunc, _ = env.step(act_in_env(agent, env, obs))
                visits[env.state_key()] += 1
                done = term or trunc
            m = max(visits.values())
            stuck.append(m >= 3)
            maxv.append(m)
            steps_stuck.append(sum(v - 1 for v in visits.values() if v >= 3))
    print(f"{variant}: epizody z krążeniem (stan >= 3 razy) {np.mean(stuck):.1%}, "
          f"średnio maks. odwiedzin {np.mean(maxv):.1f}, kroki w krążeniu {np.mean(steps_stuck):.1f}")

# Ta sama gra, w której agent krążył (watch.py, ziarno 0, model z ziarnem 8)
for run, variant, memory in RUNS:
    env = MineEnv(random_maps=True, obs_mode="hint", hint_kind="path", hint_memory=memory)
    agent = make_agent(str(model_path(run, variant, 8)), env, 0, 6)
    obs, _ = env.reset(seed=0)
    done, total, t = False, 0.0, 0
    while not done:
        obs, r, term, trunc, info = env.step(act_in_env(agent, env, obs))
        total, t, done = total + r, t + 1, term or trunc
    print(f"gra z podglądu, {variant}: wynik {total:.1f}, {t} ruchów, {info['reason']}, ładunek {info['collected_value']}")
