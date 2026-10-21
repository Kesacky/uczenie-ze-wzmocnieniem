"""Wspólny interfejs agentów.

`watch.py` i `record.py` wołają `reset()` (jeśli istnieje) na początku epizodu
i `act(obs, greedy=True)` w każdym kroku. Ten sam interfejs mają agenci
tablicowi z `miner_rl/agents/tabular.py` (Q-learning, SARSA), więc działają
w podglądzie i nagrywaniu bez zmian w rendererze.
"""

from __future__ import annotations


class Agent:
    name = "agent"

    def reset(self) -> None:
        """Wyczyść pamięć epizodu (domyślnie nic)."""

    def act(self, obs, greedy: bool = False) -> int:
        raise NotImplementedError
