"""Tabularne algorytmy RL: Q-learning (off-policy) i SARSA (on-policy).

Agenci działają na dowolnym środowisku w stylu Gymnasium z dyskretną przestrzenią
akcji. Tablica Q jest słownikiem: klucz to stan (krotka), wartość to wektor Q(s, ·).
Dzięki temu pamięć zajmują tylko stany faktycznie odwiedzone, a agent nie musi
znać rozmiaru przestrzeni stanów z góry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from pathlib import Path
from typing import Callable, Hashable

import numpy as np


def default_key(obs) -> Hashable:
    """Zamienia obserwację na klucz słownika (krotkę liczb całkowitych)."""
    if isinstance(obs, (tuple, list)):
        return tuple(int(v) for v in obs)
    if isinstance(obs, np.ndarray):
        return tuple(int(v) for v in obs.ravel())
    if isinstance(obs, dict):
        return tuple(default_key(obs[k]) if isinstance(obs[k], (tuple, list, np.ndarray)) else int(obs[k])
                     for k in sorted(obs))
    return int(obs)


@dataclass
class EpsilonSchedule:
    """Liniowo malejące epsilon: od `start` do `end` w ciągu `decay_episodes` epizodów."""
    start: float = 1.0
    end: float = 0.05
    decay_episodes: int = 1000

    def value(self, episode: int) -> float:
        if self.decay_episodes <= 0:
            return self.end
        if episode >= self.decay_episodes:
            return self.end
        frac = episode / self.decay_episodes
        return self.start + frac * (self.end - self.start)


@dataclass
class TabularAgent:
    """Wspólna część Q-learningu i SARSA: tablica Q, polityka ε-zachłanna, zapis i odczyt.

    `q_init` to wartość początkowa Q(s,a) dla nowego stanu. Wartość dodatnia
    (optymistyczna inicjalizacja) zachęca do odwiedzania nieznanych stanów: dopóki
    agent ich nie pozna, wyglądają lepiej niż natychmiastowe wyjście z kopalni.
    """
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: EpsilonSchedule = field(default_factory=EpsilonSchedule)
    q_init: float = 0.0
    seed: int | None = None
    key_fn: Callable = default_key
    q: dict = field(default_factory=dict)

    name = "tabular"

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.current_epsilon = self.epsilon.start

    # --- tablica Q -------------------------------------------------------
    def values(self, obs) -> np.ndarray:
        key = self.key_fn(obs)
        if key not in self.q:
            self.q[key] = np.full(self.n_actions, self.q_init, dtype=np.float64)
        return self.q[key]

    def greedy_action(self, obs) -> int:
        v = self.values(obs)
        best = np.flatnonzero(v == v.max())  # losowe rozstrzyganie remisów
        return int(self.rng.choice(best))

    def act(self, obs, greedy: bool = False) -> int:
        """Polityka ε-zachłanna (lub zachłanna przy ewaluacji)."""
        if not greedy and self.rng.random() < self.current_epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(obs)

    def start_episode(self, episode: int) -> None:
        self.current_epsilon = self.epsilon.value(episode)

    # --- aktualizacja (implementują podklasy) ----------------------------
    def td_target(self, reward: float, next_obs, next_action: int | None, terminated: bool) -> float:
        raise NotImplementedError

    def update(self, obs, action: int, reward: float, next_obs, next_action: int | None,
               terminated: bool) -> float:
        """Jedna aktualizacja TD. Zwraca błąd TD (do diagnostyki)."""
        target = self.td_target(reward, next_obs, next_action, terminated)
        v = self.values(obs)
        td_error = target - v[action]
        v[action] += self.alpha * td_error
        return float(td_error)

    # --- zapis i odczyt --------------------------------------------------
    def save(self, path: str | Path) -> None:
        data = {"class": type(self).__name__, "n_actions": self.n_actions, "alpha": self.alpha,
                "gamma": self.gamma, "q_init": self.q_init, "q": {k: v.tolist() for k, v in self.q.items()}}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path: str | Path) -> "TabularAgent":
        """Wczytuje tylko własne pliki modeli projektu (pickle nie jest bezpieczny dla obcych plików)."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        cls = {"QLearningAgent": QLearningAgent, "SarsaAgent": SarsaAgent}[data["class"]]
        agent = cls(n_actions=data["n_actions"], alpha=data["alpha"], gamma=data["gamma"],
                    epsilon=EpsilonSchedule(0.0, 0.0, 0), q_init=data.get("q_init", 0.0))
        agent.q = {k: np.array(v, dtype=np.float64) for k, v in data["q"].items()}
        agent.current_epsilon = 0.0
        return agent


class QLearningAgent(TabularAgent):
    """Q-learning: cel TD używa najlepszej akcji w stanie następnym (off-policy).

    Q(s,a) <- Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
    """
    name = "q_learning"

    def td_target(self, reward, next_obs, next_action, terminated):
        if terminated:
            return reward
        return reward + self.gamma * float(self.values(next_obs).max())


class SarsaAgent(TabularAgent):
    """SARSA: cel TD używa akcji faktycznie wybranej przez politykę ε-zachłanną (on-policy).

    Q(s,a) <- Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]
    Ponieważ uwzględnia losowe ruchy eksploracyjne, w środowiskach z ryzykiem
    zwykle uczy się ostrożniejszych strategii niż Q-learning.
    """
    name = "sarsa"

    def td_target(self, reward, next_obs, next_action, terminated):
        if terminated:
            return reward
        return reward + self.gamma * float(self.values(next_obs)[next_action])


def run_episode(env, agent: TabularAgent, learn: bool, seed: int | None = None,
                greedy: bool = False, max_steps: int = 10_000) -> dict:
    """Rozgrywa jeden epizod. Przy learn=True aktualizuje tablicę Q.

    Rozróżnia `terminated` (prawdziwy koniec: brak bootstrapu) od `truncated`
    (obcięcie limitem kroków: cel TD nadal szacuje wartość stanu następnego).
    """
    obs, info = env.reset(seed=seed)
    action = agent.act(obs, greedy=greedy)
    total, steps, max_depth = 0.0, 0, 0
    while True:
        next_obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        max_depth = max(max_depth, info.get("depth", 0))
        done = terminated or truncated or steps >= max_steps
        next_action = None if terminated else agent.act(next_obs, greedy=greedy)
        if learn:
            agent.update(obs, action, reward, next_obs, next_action, terminated)
        if done:
            break
        obs, action = next_obs, next_action
    return {"return": total, "length": steps, "terminated": bool(terminated), "max_depth": max_depth,
            "info": info}


def train(env, agent: TabularAgent, episodes: int, seed: int = 0,
          callback: Callable[[int, dict], None] | None = None) -> list[dict]:
    """Trening przez `episodes` epizodów. Ziarno epizodu i to `seed + i`, więc całość jest powtarzalna."""
    history = []
    for i in range(episodes):
        agent.start_episode(i)
        result = run_episode(env, agent, learn=True, seed=seed + i)
        result["episode"] = i
        result["epsilon"] = agent.current_epsilon
        history.append(result)
        if callback:
            callback(i, result)
    return history


def evaluate(env, agent: TabularAgent, episodes: int, seed: int = 1_000_000) -> list[dict]:
    """Ewaluacja polityką zachłanną, bez uczenia, na ziarnach rozłącznych z treningiem."""
    saved = agent.current_epsilon
    agent.current_epsilon = 0.0
    try:
        return [run_episode(env, agent, learn=False, seed=seed + i, greedy=True) for i in range(episodes)]
    finally:
        agent.current_epsilon = saved
