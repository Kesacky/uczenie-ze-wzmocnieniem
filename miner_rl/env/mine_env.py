"""Środowisko „Górnik w niestabilnej kopalni” z API Gymnasium.

Implementuje formalizację MDP z raportu (`report/main.tex`, sekcja „Formalizacja RL/MDP”).

Obserwacja (stan) to wektor liczb całkowitych w kolejności z raportu:
    s = (x, y, sigma, o, m_1..m_K, n)
x, y - pozycja górnika; sigma - stabilność stropu 0..5; o - olej w przedziałach
0..3; m - maska wydobytych złóż (K = liczba złóż na mapie, K <= 5); n - liczba
stempli 0..2. Ładunek nie jest częścią stanu: to suma wartości złóż z maski.

Akcje: 0 góra, 1 dół, 2 lewo, 3 prawo, 4 podepnij.
Wejście na niewydobyte złoże jest jego wydobyciem. Akcja bez skutku (ruch w skałę
lub poza mapę, podepnij bez stempli lub przy pełnej stabilności) jest niedozwolona:
stan się nie zmienia (poza upływem czasu), a nagroda zawiera dodatkową karę.

Kolejność zdarzeń w kroku `step(a)`:
 1. kara za krok;
 2. wykonanie akcji (ruch, wydobycie albo podparcie); kara za akcję niedozwoloną;
 3. wejście na wyjście E kończy epizod nagrodą równą wartości ładunku;
 4. zdarzenie korytarza: gdy liczba wydobytych złóż osiągnie N, pola `X` stają się
    skałą (zależy tylko od maski, więc stan pozostaje Markowowski);
 5. zużycie oleju;
 6. losowanie zawału z p = p0 * (5 - sigma) * (1 + y / H) -> kara, ładunek przepada;
 7. brak oleju -> kara, ładunek przepada;
 8. limit kroków ustawia `truncated`.
Liczba losowa do zawału jest pobierana w każdym kroku (także gdy p = 0 lub po
wyjściu), więc przy tym samym ziarnie i tych samych akcjach przebieg jest identyczny.

Moduł nie importuje pygame. Renderer czyta publiczne atrybuty (`grid`, `pos`,
`facing`, `stability`, `oil`, `load_value`, `props`, `prop_cells`, `last_dug`,
`last_info`, `event_fired`, ...), nigdy ich nie zmienia. `facing` (kierunek
ostatniego udanego ruchu) służy tylko grafice i nie należy do stanu.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from miner_rl.config import GameConfig
from miner_rl.env.maps import DEPOSITS, ROCK, WALKABLE, MineMap, load_map

UP, DOWN, LEFT, RIGHT, PROP = range(5)
ACTION_NAMES = ("gora", "dol", "lewo", "prawo", "podepnij")
DIRECTIONS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

REASON_EXIT = "exit"
REASON_COLLAPSE = "collapse"
REASON_OIL = "out_of_oil"
REASON_LIMIT = "step_limit"


class State(NamedTuple):
    """Czytelny widok obserwacji (por. `MineEnv.decode`)."""
    x: int
    y: int
    stability: int
    oil: int
    mined: tuple[int, ...]
    props: int


class MineEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 8}

    def __init__(
        self,
        mine_map: str | Path | MineMap = "m1_stabilna",
        config: GameConfig | None = None,
        render_mode: str | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        self.map = mine_map if isinstance(mine_map, MineMap) else load_map(mine_map)
        base = config or GameConfig()
        # domyślne < nagłówek mapy < jawne nadpisanie
        self.config = base.replace(**self.map.overrides).replace(**overrides)
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Nieobsługiwany render_mode: {render_mode}")
        self.render_mode = render_mode

        cfg = self.config
        self.width, self.height = self.map.width, self.map.height
        self.deposits = self.map.deposits
        self.n_deposits = len(self.deposits)
        self._deposit_index = {(x, y): i for i, (x, y, _) in enumerate(self.deposits)}
        self._deposit_value = [cfg.deposit_values[ch] for _, _, ch in self.deposits] \
            if all(ch in cfg.deposit_values for _, _, ch in self.deposits) else None
        if self._deposit_value is None:
            raise ValueError("Brak wartości dla części złóż w deposit_values")
        if self.n_deposits > cfg.max_deposits:
            raise ValueError(f"{self.map.name}: {self.n_deposits} złóż, dozwolone najwyżej {cfg.max_deposits}")
        if not 0 <= cfg.stability_start <= cfg.stability_max:
            raise ValueError("stability_start poza zakresem 0..stability_max")
        if not 0 <= cfg.props_start <= cfg.props_max:
            raise ValueError("props_start poza zakresem 0..props_max")
        self.exit_pos = self.map.exit

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.MultiDiscrete(
            [self.width, self.height, cfg.stability_max + 1, cfg.oil_bins]
            + [2] * self.n_deposits + [cfg.props_max + 1]
        )
        self._needs_reset = True
        self._init_state()

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._init_state()
        self._needs_reset = False
        info = self._info(reason=None, collapsed=False, illegal=False, event=None, p=0.0)
        self.last_info = info
        return self._obs(), info

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Epizod zakończony lub nierozpoczęty: wywołaj reset().")
        if not self.action_space.contains(action):
            raise ValueError(f"Niepoprawna akcja: {action}")
        action = int(action)
        cfg = self.config
        self.steps += 1
        self.last_action = action
        self.last_dug = None
        reward = cfg.reward_step
        legal = self._apply(action)
        if not legal:
            reward += cfg.reward_illegal

        terminated = False
        reason = None
        collapsed = False
        event = None
        p = 0.0
        draw = self.np_random.random()     # zawsze, dla stałego strumienia losowań

        if self.pos == self.exit_pos:
            reward += cfg.reward_exit_multiplier * self.load_value
            terminated, reason = True, REASON_EXIT
        else:
            event = self._corridor_event()
            self.oil = max(0, self.oil - cfg.oil_per_step)
            p = self.collapse_probability()
            if draw < p:
                reward += cfg.reward_collapse
                terminated, reason, collapsed = True, REASON_COLLAPSE, True
                self.lost_value = self.load_value
            elif self.oil <= 0:
                reward += cfg.reward_out_of_oil
                terminated, reason = True, REASON_OIL
                self.lost_value = self.load_value

        truncated = not terminated and self.steps >= cfg.step_limit
        if truncated:
            reason = REASON_LIMIT
        self._needs_reset = terminated or truncated
        self.total_reward += reward
        info = self._info(reason=reason, collapsed=collapsed, illegal=not legal, event=event, p=p)
        self.last_info = info
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self.to_text()
        return None

    # -------------------------------------------------------------- stan
    def _init_state(self) -> None:
        cfg = self.config
        self.grid: list[list[str]] = [list(row) for row in self.map.rows]
        self.pos: tuple[int, int] = self.map.start
        self.facing: int = DOWN
        self.stability: int = cfg.stability_start
        self.props: int = cfg.props_start
        self.oil: int = cfg.oil_start
        self.load_value: int = 0
        self.lost_value: int = 0
        self.mined = np.zeros(self.n_deposits, dtype=np.int64)
        self.steps: int = 0
        self.total_reward: float = 0.0
        self.event_fired: bool = False
        self.prop_cells: list[tuple[int, int]] = []
        self.last_action: int | None = None
        self.last_dug: tuple[int, int] | None = None
        self.last_info: dict[str, Any] = {}

    def _apply(self, action: int) -> bool:
        """Wykonaj akcję; zwraca, czy była dozwolona."""
        cfg = self.config
        if action in DIRECTIONS:
            dx, dy = DIRECTIONS[action]
            tx, ty = self.pos[0] + dx, self.pos[1] + dy
            if not (0 <= tx < self.width and 0 <= ty < self.height):
                return False
            target = self.grid[ty][tx]
            if target in DEPOSITS:                       # wydobycie = wejście na złoże
                index = self._deposit_index[(tx, ty)]
                self.mined[index] = 1
                self.load_value += self._deposit_value[index]
                self.grid[ty][tx] = "."
                self.stability = max(0, self.stability - cfg.dig_stability_cost)
                self.last_dug = (tx, ty)
            elif target not in WALKABLE:                 # lita skała
                return False
            self.pos = (tx, ty)
            self.facing = action
            return True

        # PROP
        if self.props <= 0 or self.stability >= cfg.stability_max:
            return False
        self.props -= 1
        self.stability = min(cfg.stability_max, self.stability + cfg.prop_stability_gain)
        self.prop_cells.append(self.pos)
        return True

    def _corridor_event(self) -> str | None:
        n = self.config.corridor_event_after
        if self.event_fired or n <= 0 or int(self.mined.sum()) < n:
            return None
        self.event_fired = True
        cells = self.map.collapse_cells
        for x, y in cells:
            # Górnik nie może stać na X w tym kroku: zdarzenie wywołuje wydobycie,
            # a wydobycie oznacza wejście na pole złoża.
            self.grid[y][x] = ROCK
        return "corridor_collapse" if cells else None

    def collapse_probability(self, stability: int | None = None, y: int | None = None) -> float:
        cfg = self.config
        stability = self.stability if stability is None else stability
        y = self.pos[1] if y is None else y
        if cfg.collapse_prob is not None:
            p = cfg.collapse_prob
        else:
            p = cfg.collapse_base * (cfg.stability_max - stability) * (1.0 + y / self.height)
        return float(min(max(p, 0.0), 1.0))

    # ---------------------------------------------------------- obserwacja
    def oil_bin(self, oil: int | None = None) -> int:
        """Przedział oleju 0..oil_bins-1 (przy 100 i 4 przedziałach: 0-24, 25-49, 50-74, 75-100)."""
        cfg = self.config
        oil = self.oil if oil is None else oil
        oil = min(max(oil, 0), cfg.oil_start)
        return min(cfg.oil_bins - 1, oil * cfg.oil_bins // max(cfg.oil_start, 1))

    def _obs(self) -> np.ndarray:
        return np.array(
            [self.pos[0], self.pos[1], self.stability, self.oil_bin(), *self.mined, self.props],
            dtype=np.int64,
        )

    def decode(self, obs) -> State:
        """Rozpakowanie obserwacji na nazwane pola."""
        values = [int(v) for v in np.asarray(obs).ravel()]
        k = self.n_deposits
        return State(values[0], values[1], values[2], values[3], tuple(values[4:4 + k]), values[4 + k])

    def load_from_mask(self, mined) -> int:
        """Wartość ładunku wyznaczona z maski złóż (ładunek nie jest osobną zmienną stanu)."""
        return int(sum(v for v, m in zip(self._deposit_value, mined) if m))

    def _info(self, *, reason, collapsed, illegal, event, p) -> dict[str, Any]:
        return {
            "reason": reason,
            "collapsed": collapsed,
            "illegal": illegal,
            "event": event,
            "collected": int(self.mined.sum()),
            "collected_value": int(self.load_value),
            "lost_value": int(self.lost_value),
            "steps": self.steps,
            "oil": int(self.oil),
            "stability": int(self.stability),
            "depth": int(self.pos[1]),
            "collapse_prob": float(p),
            "action": None if self.last_action is None else ACTION_NAMES[self.last_action],
        }

    @property
    def n_states(self) -> int:
        """Liczba kombinacji stanu (górne ograniczenie; osiągalna jest tylko część)."""
        return int(np.prod(self.observation_space.nvec, dtype=np.int64))

    def obs_to_key(self, obs) -> tuple[int, ...]:
        """Krotka stanu, np. klucz słownikowej tablicy Q."""
        return tuple(int(v) for v in np.asarray(obs).ravel())

    def obs_to_index(self, obs) -> int:
        """Jednoznaczny indeks 0..n_states-1 (system mieszanych podstaw)."""
        index = 0
        for value, radix in zip(self.obs_to_key(obs), self.observation_space.nvec):
            index = index * int(radix) + value
        return index

    def obs_to_vector(self, obs) -> np.ndarray:
        """Wektor cech w [0, 1] (np. dla DQN), długość 5 + K."""
        nvec = self.observation_space.nvec
        values = np.asarray(obs, dtype=np.float32).ravel()
        return values / np.maximum(nvec - 1, 1).astype(np.float32)

    @property
    def vector_size(self) -> int:
        return 5 + self.n_deposits

    # ------------------------------------------------------------ pomocnicze
    def to_text(self) -> str:
        rows = ["".join(r) for r in self.grid]
        x, y = self.pos
        rows[y] = rows[y][:x] + "@" + rows[y][x + 1:]
        hud = (f"krok {self.steps}  stabilnosc {self.stability}/{self.config.stability_max}  "
               f"olej {self.oil}  ladunek {self.load_value}  stemple {self.props}")
        return "\n".join(rows + [hud])
