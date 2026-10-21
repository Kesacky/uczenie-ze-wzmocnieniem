"""Agenci bazowi: losowy i progowy („kop do progu ładunku X, potem wracaj”)."""

from __future__ import annotations

import heapq

import numpy as np

from miner_rl.agents.base import Agent
from miner_rl.config import GameConfig
from miner_rl.env.maps import DEPOSITS, ROCK, WALKABLE, MineMap
from miner_rl.env.mine_env import DIRECTIONS, PROP


class RandomAgent(Agent):
    """Akcja z rozkładu jednostajnego; własny generator dla powtarzalności."""

    name = "random"

    def __init__(self, n_actions: int = 5, seed: int | None = None) -> None:
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs, greedy: bool = False) -> int:
        return int(self.rng.integers(self.n_actions))


class ThresholdAgent(Agent):
    """Heurystyka z pełną wiedzą o mapie, ale bez dostępu do wnętrza środowiska.

    Stan mapy odtwarza z obserwacji: maska mówi, które złoża wydobyto, a liczba
    wydobytych złóż, czy pola `X` zamieniły się już w skałę.
    Reguły (w tej kolejności):
      1. stabilność <= `prop_at` i są stemple -> `podepnij`;
      2. ładunek >= `threshold` albo olej w przedziale <= `oil_return_bin`
         -> najkrótsza droga do wyjścia;
      3. w przeciwnym razie najbliższe osiągalne niewydobyte złoże
         (wejście = wydobycie); gdy żadnego nie ma -> wyjście.
    Ścieżki omijają inne złoża (koszt `deposit_cost`), żeby nie wydobywać ich
    przypadkiem; przechodzą przez nie tylko wtedy, gdy nie ma innej drogi.
    """

    name = "threshold"

    def __init__(self, mine_map: MineMap, config: GameConfig, threshold: float = 6,
                 prop_at: int = 1, oil_return_bin: int = 0, deposit_cost: int = 50) -> None:
        self.map = mine_map
        self.config = config   # konfiguracja już scalona przez MineEnv (env.config)
        self.threshold = threshold
        self.prop_at = prop_at
        self.oil_return_bin = oil_return_bin
        self.deposit_cost = deposit_cost
        self.deposits = mine_map.deposits
        self.k = len(self.deposits)
        self.values = [config.deposit_values[ch] for _, _, ch in self.deposits]
        self.name = f"threshold_{threshold:g}"

    @classmethod
    def for_env(cls, env, threshold: float = 6, **kwargs) -> "ThresholdAgent":
        return cls(env.map, env.config, threshold=threshold, **kwargs)

    # ----------------------------------------------------------- mapa
    def _grid(self, mined) -> list[list[str]]:
        grid = [list(row) for row in self.map.rows]
        for i, (x, y, _) in enumerate(self.deposits):
            if mined[i]:
                grid[y][x] = "."
        n = self.config.corridor_event_after
        if n > 0 and sum(mined) >= n:
            for x, y in self.map.collapse_cells:
                grid[y][x] = ROCK
        return grid

    def _dijkstra(self, grid, start, target=None):
        """Koszt i pierwszy ruch do każdego pola; `target` (złoże) ma koszt zwykły."""
        height, width = len(grid), len(grid[0])
        best = {start: (0, None)}
        heap = [(0, 0, start, None)]
        counter = 0
        while heap:
            cost, _, cell, first = heapq.heappop(heap)
            if cost > best[cell][0]:
                continue
            x, y = cell
            for action, (dx, dy) in DIRECTIONS.items():
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                ch = grid[ny][nx]
                if ch in WALKABLE or (nx, ny) == target:
                    step = 1
                elif ch in DEPOSITS:
                    step = self.deposit_cost
                else:
                    continue
                new_cost = cost + step
                if new_cost < best.get((nx, ny), (float("inf"),))[0]:
                    move = first if first is not None else action
                    best[(nx, ny)] = (new_cost, move)
                    counter += 1
                    heapq.heappush(heap, (new_cost, counter, (nx, ny), move))
        return best

    # ---------------------------------------------------------- decyzja
    def act(self, obs, greedy: bool = False) -> int:
        values = [int(v) for v in np.asarray(obs).ravel()]
        pos = (values[0], values[1])
        stability, oil_bin = values[2], values[3]
        mined = values[4:4 + self.k]
        props = values[4 + self.k]
        load = sum(v for v, m in zip(self.values, mined) if m)

        if stability <= self.prop_at and props > 0 and stability < self.config.stability_max:
            return PROP

        grid = self._grid(mined)
        if load < self.threshold and oil_bin > self.oil_return_bin:
            options = []
            for i, (x, y, _) in enumerate(self.deposits):
                if not mined[i]:
                    reach = self._dijkstra(grid, pos, target=(x, y))
                    if (x, y) in reach:
                        options.append((reach[(x, y)][0], i, reach[(x, y)][1]))
            if options:
                return int(min(options)[2])
        reach = self._dijkstra(grid, pos)
        if self.map.exit in reach and reach[self.map.exit][1] is not None:
            return int(reach[self.map.exit][1])
        return PROP   # brak drogi do wyjścia (nie powinno wystąpić na mapach projektu)
