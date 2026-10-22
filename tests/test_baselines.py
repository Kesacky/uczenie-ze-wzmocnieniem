"""Agenci bazowi kończą epizody bez wyjątków i zachowują się sensownie."""

import pytest

from miner_rl.agents import RandomAgent, ThresholdAgent
from miner_rl.env import MineEnv, list_maps

MAPS = [p.stem for p in list_maps()]


def run(env, agent, episodes, seed0=0):
    infos = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        agent.reset()
        done = False
        while not done:
            action = agent.act(obs, greedy=True)
            assert env.action_space.contains(action)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        infos.append(info)
    return infos


@pytest.mark.parametrize("name", MAPS)
def test_random_agent_100_episodes(name):
    infos = run(MineEnv(name), RandomAgent(seed=0), 100)
    assert len(infos) == 100 and all(i["reason"] for i in infos)


@pytest.mark.parametrize("name", MAPS)
def test_threshold_agent_100_episodes(name):
    env = MineEnv(name)
    infos = run(env, ThresholdAgent.for_env(env, threshold=6), 100)
    assert all(i["reason"] in {"exit", "collapse", "out_of_oil", "step_limit"} for i in infos)


@pytest.mark.parametrize("name", MAPS)
def test_threshold_agent_collects_everything_without_risk(name):
    # Bez zawałów agent bez progu wydobywa wszystkie złoża i wychodzi, bez akcji niedozwolonych.
    env = MineEnv(name, collapse_prob=0.0, oil_start=1000, step_limit=1000)
    agent = ThresholdAgent.for_env(env, threshold=float("inf"))
    obs, _ = env.reset(seed=0)
    done, illegal = False, 0
    while not done:
        obs, _, term, trunc, info = env.step(agent.act(obs))
        illegal += info["illegal"]
        done = term or trunc
    assert info["reason"] == "exit" and info["collected"] == env.n_deposits and illegal == 0


def test_threshold_agent_respects_threshold():
    env = MineEnv("m1_stabilna", collapse_prob=0.0)
    info = run(env, ThresholdAgent.for_env(env, threshold=1), 1)[0]
    assert info["reason"] == "exit" and info["collected"] == 1


def test_threshold_agent_props_at_low_stability():
    env = MineEnv("m1_stabilna", stability_start=1, collapse_prob=0.0)
    obs, _ = env.reset(seed=0)
    assert ThresholdAgent.for_env(env).act(obs) == 4


def test_threshold_agent_on_random_maps():
    env = MineEnv(random_maps=True, collapse_prob=0.0, oil_start=1000, step_limit=1000)
    agent = ThresholdAgent.for_env(env, threshold=float("inf"))
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        agent.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(agent.act(obs))
            done = term or trunc
        assert info["reason"] == "exit" and info["collected"] == env.n_deposits
