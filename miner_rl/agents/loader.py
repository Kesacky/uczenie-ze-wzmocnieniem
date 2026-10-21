"""Tworzenie agentów dla watch.py i record.py na podstawie tekstowej specyfikacji.

- "random" -> RandomAgent, "threshold" -> ThresholdAgent (próg z argumentu),
- ścieżka do pliku .pkl -> model zapisany przez `TabularAgent.save` (Q-learning, SARSA)
  albo dowolny zapiklowany obiekt z metodą `act(obs, greedy=...)`,
- "pakiet.moduł:funkcja" -> wynik `funkcja(env)` (punkt zaczepienia dla innych agentów).
Pliki pickle wczytuj tylko z zaufanego źródła (własne modele projektu).
"""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path

from .baselines import RandomAgent, ThresholdAgent


def make_agent(spec: str, env, seed=None, threshold=6.0):
    if spec == "random":
        return RandomAgent(n_actions=env.action_space.n, seed=seed)
    if spec == "threshold":
        return ThresholdAgent.for_env(env, threshold)
    if spec.endswith(".pkl"):
        with Path(spec).open("rb") as stream:
            data = pickle.load(stream)
        if isinstance(data, dict) and "class" in data and "q" in data:
            from .tabular import TabularAgent   # format TabularAgent.save
            agent = TabularAgent.load(spec)
        else:
            agent = data
    elif ":" in spec:
        module, function = spec.split(":", 1)
        try:
            agent = getattr(importlib.import_module(module), function)(env)
        except (ImportError, AttributeError, TypeError) as error:
            raise ValueError(f"Nie można utworzyć agenta: {spec}") from error
    else:
        raise ValueError(f"Nieznana specyfikacja agenta: {spec}")
    if not callable(getattr(agent, "act", None)):
        raise ValueError(f"Wczytany obiekt nie jest agentem (brak metody act): {spec}")
    return agent
