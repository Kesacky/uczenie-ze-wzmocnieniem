"""Testy agentów tablicowych na klasycznym zadaniu Cliff Walking (Sutton i Barto, przykład 6.6).

Mapa 4×12: start w lewym dolnym rogu, cel w prawym dolnym, między nimi urwisko
(kara -100 i powrót na start). Znany wynik: przy stałym ε=0,1 Q-learning uczy się
najkrótszej drogi wzdłuż urwiska, a SARSA bezpieczniejszej, dłuższej drogi.
To ten sam efekt, którego oczekujemy w kopalni (ryzyko zawału).
"""
import numpy as np
import pytest

from miner_rl.agents.tabular import (EpsilonSchedule, QLearningAgent, SarsaAgent,
                                     TabularAgent, evaluate, train)


class CliffWalk:
    H, W = 4, 12
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def reset(self, seed=None):
        self.pos = (self.H - 1, 0)
        self.t = 0
        return self.pos, {}

    def step(self, a):
        dr, dc = self.MOVES[a]
        r, c = self.pos
        r = min(max(r + dr, 0), self.H - 1)
        c = min(max(c + dc, 0), self.W - 1)
        self.t += 1
        if r == self.H - 1 and 0 < c < self.W - 1:  # urwisko
            self.pos = (self.H - 1, 0)
            return self.pos, -100.0, False, self.t >= 500, {"fell": True}
        self.pos = (r, c)
        done = self.pos == (self.H - 1, self.W - 1)
        return self.pos, -1.0, done, self.t >= 500, {}


def greedy_path_length(agent):
    result = evaluate(CliffWalk(), agent, episodes=1)[0]
    assert result["terminated"], "polityka zachłanna nie dochodzi do celu"
    return result["length"]


def make(cls, seed):
    return cls(n_actions=4, alpha=0.2, gamma=1.0, epsilon=EpsilonSchedule(0.1, 0.1, 0), seed=seed)


def test_q_learning_finds_shortest_path():
    agent = make(QLearningAgent, 0)
    train(CliffWalk(), agent, episodes=1000, seed=0)
    assert greedy_path_length(agent) == 13


def test_sarsa_is_more_cautious_than_q_learning():
    q_len, s_len = [], []
    for seed in range(3):
        q = make(QLearningAgent, seed); train(CliffWalk(), q, 1000, seed)
        s = make(SarsaAgent, seed); train(CliffWalk(), s, 1000, seed)
        q_len.append(greedy_path_length(q)); s_len.append(greedy_path_length(s))
    assert np.mean(s_len) > np.mean(q_len)


def test_training_is_reproducible():
    a, b = make(QLearningAgent, 7), make(QLearningAgent, 7)
    ha = train(CliffWalk(), a, 50, seed=7)
    hb = train(CliffWalk(), b, 50, seed=7)
    assert [h["return"] for h in ha] == [h["return"] for h in hb]


def test_terminal_update_does_not_bootstrap():
    agent = make(QLearningAgent, 0)
    agent.values((9, 9))[:] = 1000.0
    agent.update((0, 0), 1, 5.0, (9, 9), None, terminated=True)
    assert agent.values((0, 0))[1] == 0.2 * 5.0


def test_epsilon_schedule():
    s = EpsilonSchedule(1.0, 0.1, 100)
    assert s.value(0) == 1.0 and abs(s.value(50) - 0.55) < 1e-9 and s.value(500) == 0.1


def test_save_and_load(tmp_path):
    agent = make(SarsaAgent, 0)
    train(CliffWalk(), agent, 1000, seed=0)
    path = tmp_path / "sarsa.pkl"
    agent.save(path)
    loaded = TabularAgent.load(path)
    assert isinstance(loaded, SarsaAgent)
    assert set(loaded.q) == set(agent.q)
    assert all(np.array_equal(loaded.q[k], agent.q[k]) for k in agent.q)
    assert greedy_path_length(loaded) == greedy_path_length(agent)


def test_optimistic_init_and_save_load(tmp_path):
    agent = QLearningAgent(n_actions=4, q_init=5.0, seed=0)
    assert list(agent.values((1, 2))) == [5.0] * 4
    path = tmp_path / "q.pkl"
    agent.save(path)
    loaded = TabularAgent.load(path)
    assert loaded.q_init == 5.0 and list(loaded.values((7, 7))) == [5.0] * 4


def test_masked_agent_never_takes_illegal_actions():
    from miner_rl.env import MineEnv
    env = MineEnv("m1_stabilna")
    agent = SarsaAgent(n_actions=5, q_init=30.0, mask_illegal=True,
                       epsilon=EpsilonSchedule(1.0, 0.05, 150), seed=0)
    illegal = []
    for i in range(200):
        agent.start_episode(i)
        obs, _ = env.reset(seed=i)
        action, done = agent.act(obs, mask=env.action_mask()), False
        while not done:
            obs2, r, term, trunc, info = env.step(action)
            illegal.append(info["illegal"])
            mask = None if term else env.action_mask()
            action2 = None if term else agent.act(obs2, mask=mask)
            agent.update(obs, action, r, obs2, action2, term, mask)
            obs, action, done = obs2, action2, term or trunc
    assert not any(illegal)


def test_q_learning_target_uses_only_legal_actions():
    agent = QLearningAgent(n_actions=3, alpha=1.0, gamma=1.0, mask_illegal=True)
    agent.values((1,))[:] = [100.0, 2.0, 3.0]          # akcja 0 niedozwolona
    agent.update((0,), 0, 0.0, (1,), None, False, np.array([False, True, True]))
    assert agent.values((0,))[0] == 3.0


def test_count_bonus_decreases_with_visits():
    agent = QLearningAgent(n_actions=2, alpha=1.0, gamma=0.0, bonus_beta=4.0)
    agent.update((0,), 1, 0.0, (1,), None, True)
    assert agent.values((0,))[1] == 4.0                 # β/√1
    agent.update((0,), 1, 0.0, (1,), None, True)
    assert agent.values((0,))[1] == pytest.approx(4.0 / np.sqrt(2))   # β/√2, α = 1
