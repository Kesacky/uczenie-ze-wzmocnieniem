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
for run, ho in [("narrator_losowe", 2), ("narrator_pamiec", 3)]:
    stuck, maxv, steps_stuck = [], [], []
    for seed in range(10):
        model = ROOT / f"results/train/{run}/models/losowa_q_learning_q30_n100000_ho{ho}_s{seed}.pkl"
        env = MineEnv(random_maps=True, obs_mode="hint", hint_kind="path", hint_memory=ho == 3)
        agent = make_agent(str(model), env, 0, 6)
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
    print(f"ho{ho}: epizody z krążeniem (stan >= 3 razy) {np.mean(stuck):.1%}, "
          f"średnio maks. odwiedzin {np.mean(maxv):.1f}, kroki w krążeniu {np.mean(steps_stuck):.1f}")

# Ta sama gra, w której agent krążył (watch.py, ziarno 0, model z ziarnem 8)
for run, ho in [("narrator_losowe", 2), ("narrator_pamiec", 3)]:
    env = MineEnv(random_maps=True, obs_mode="hint", hint_kind="path", hint_memory=ho == 3)
    agent = make_agent(str(ROOT / f"results/train/{run}/models/losowa_q_learning_q30_n100000_ho{ho}_s8.pkl"), env, 0, 6)
    obs, _ = env.reset(seed=0)
    done, total, t = False, 0.0, 0
    while not done:
        obs, r, term, trunc, info = env.step(act_in_env(agent, env, obs))
        total, t, done = total + r, t + 1, term or trunc
    print(f"gra z podglądu, ho{ho}: wynik {total:.1f}, {t} ruchów, {info['reason']}, ładunek {info['collected_value']}")
