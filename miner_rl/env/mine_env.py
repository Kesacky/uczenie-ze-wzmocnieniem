"""Środowisko „Górnik w niestabilnej kopalni” z API Gymnasium.

Implementuje formalizację MDP z raportu (`report/main.tex`, sekcja „Formalizacja RL/MDP”).

Dwa warianty gry:
- stała mapa: `MineEnv("m1_stabilna")` (plik z `maps/` albo obiekt `MineMap`);
- losowa mapa: `MineEnv(random_maps=True)` – przy każdym `reset(seed)` nowa kopalnia
  z generatora (`miner_rl/env/generator.py`), wyznaczona przez ziarno epizodu
  (albo przez `options={"map_seed": s}`, gdy mapa ma być niezależna od ziarna epizodu).

Obserwacja (stan) to wektor liczb całkowitych w kolejności z raportu:
    s = (x, y, sigma, o, m_1..m_K, n)
Przy `obs_mode="local"` do stanu dołączany jest widok otoczenia górnika: kody pól
w kwadracie (2r+1)×(2r+1) wierszami (0 skała/poza mapą, 1 korytarz/start,
2 niewydobyte złoże, 3 wyjście, 4 pole X). Ten wariant jest potrzebny agentom
uczonym na losowych mapach, bo sama pozycja nie mówi nic o układzie nowej kopalni.
Przy `obs_mode="hint"` (obserwacja z podpowiedzią „narratora”, wariant dodatkowy)
obserwacja jest inna: (sigma, o, n, ładunek w przedziałach, 4 sąsiednie pola,
kierunek / odległość / rodzaj najbliższego niewydobytego złoża, kierunek / odległość
do wyjścia) - bez pozycji i maski złóż, żeby liczba stanów była mała także na
mapach losowych. Kierunek to przy `hint_kind="compass"` kierunek celu „na przełaj”
(8 stron świata), a przy `hint_kind="path"` kierunek pierwszego kroku najkrótszej
drogi korytarzami (jak w nawigacji). Przy `hint_memory=1` obserwacja ma na końcu
4 bity pamięci: czy sąsiednie pole (góra, dół, lewo, prawo) było już odwiedzone w tym
epizodzie (bez nich agent nie odróżnia powrotu od nowego miejsca i może krążyć),
a przy `hint_memory=2` liczbę wejść na te pola w przedziałach 0 / 1 / 2 i więcej.
Narrator opisuje bieżący stan kopalni; o ruchu, wydobyciu, podparciu i powrocie
decyduje agent.
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
 6. losowanie zawału z p = p_min + p0 * (5 - sigma)^k * (1 + y / H) -> kara, ładunek przepada;
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
from miner_rl.env.generator import GeneratorConfig, generate_map
from miner_rl.env.maps import COLLAPSING, DEPOSITS, EXIT, ROCK, WALKABLE, MineMap, load_map

UP, DOWN, LEFT, RIGHT, PROP = range(5)
ACTION_NAMES = ("gora", "dol", "lewo", "prawo", "podepnij")
DIRECTIONS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

REASON_EXIT = "exit"
REASON_COLLAPSE = "collapse"
REASON_OIL = "out_of_oil"
REASON_LIMIT = "step_limit"
OBS_MODES = ("state", "local", "hint")
# Obserwacja z podpowiedzią: przedziały ładunku i odległości po korytarzach (górne granice).
LOAD_BINS = (0, 9, 29, 59)            # 0 | 1-9 | 10-29 | 30-59 | 60+
DIST_BINS = (1, 3, 7)                 # 1 | 2-3 | 4-7 | 8+
# Kierunek celu: 0 = brak celu, 1..8 = (znak dx, znak dy) bez (0, 0).
HINT_KINDS = ("compass", "path")
COMPASS = {(0, -1): 1, (1, -1): 2, (1, 0): 3, (1, 1): 4, (0, 1): 5, (-1, 1): 6, (-1, 0): 7, (-1, -1): 8}


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
        mine_map: str | Path | MineMap | None = "m1_stabilna",
        config: GameConfig | None = None,
        render_mode: str | None = None,
        *,
        random_maps: bool = False,
        generator: GeneratorConfig | None = None,
        obs_mode: str = "state",
        view_radius: int = 2,
        hint_kind: str = "compass",
        hint_memory: int = 0,
        **overrides: Any,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Nieobsługiwany render_mode: {render_mode}")
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode musi być jednym z {OBS_MODES}")
        self.render_mode = render_mode
        self.random_maps = random_maps
        self.generator = generator or GeneratorConfig()
        if hint_kind not in HINT_KINDS:
            raise ValueError(f"hint_kind musi być jednym z {HINT_KINDS}")
        self.obs_mode = obs_mode
        self.view_radius = view_radius
        self.hint_kind = hint_kind
        if hint_memory not in (0, 1, 2):
            raise ValueError("hint_memory musi być 0, 1 lub 2")
        self.hint_memory = int(hint_memory)
        self._base_config = config or GameConfig()
        self._overrides = overrides
        if random_maps:
            # Mapa startowa tylko po to, by znać wymiary; reset() wylosuje właściwą.
            first = generate_map(np.random.default_rng(0), self.generator)
        else:
            first = mine_map if isinstance(mine_map, MineMap) else load_map(mine_map)
        self._load_map(first)

        cfg = self.config
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        nvec = [self.width, self.height, cfg.stability_max + 1, cfg.oil_bins] \
            + [2] * self.n_deposits + [cfg.props_max + 1]
        if obs_mode == "local":
            nvec += [5] * (2 * view_radius + 1) ** 2
        elif obs_mode == "hint":
            nvec = [cfg.stability_max + 1, cfg.oil_bins, cfg.props_max + 1, len(LOAD_BINS) + 1] + [5] * 4 \
                + [9, len(DIST_BINS) + 1, len(DEPOSITS) + 1, 9, len(DIST_BINS) + 1]
            if hint_memory:
                nvec += [int(hint_memory) + 1] * 4
        self.observation_space = spaces.MultiDiscrete(nvec)
        self._needs_reset = True
        self._init_state()

    def _load_map(self, mine_map: MineMap) -> None:
        """Ustaw mapę i konfigurację (domyślne < nagłówek mapy < jawne nadpisania)."""
        self.map = mine_map
        self.config = self._base_config.replace(**mine_map.overrides).replace(**self._overrides)
        cfg = self.config
        if hasattr(self, "width") and (mine_map.width, mine_map.height, len(mine_map.deposits)) \
                != (self.width, self.height, self.n_deposits):
            raise ValueError("Nowa mapa musi mieć te same wymiary i liczbę złóż")
        self.width, self.height = mine_map.width, mine_map.height
        self.deposits = mine_map.deposits
        self.n_deposits = len(self.deposits)
        self._deposit_index = {(x, y): i for i, (x, y, _) in enumerate(self.deposits)}
        if any(ch not in cfg.deposit_values for _, _, ch in self.deposits):
            raise ValueError("Brak wartości dla części złóż w deposit_values")
        self._deposit_value = [cfg.deposit_values[ch] for _, _, ch in self.deposits]
        if self.n_deposits > cfg.max_deposits:
            raise ValueError(f"{mine_map.name}: {self.n_deposits} złóż, dozwolone najwyżej {cfg.max_deposits}")
        if not 0 <= cfg.stability_start <= cfg.stability_max:
            raise ValueError("stability_start poza zakresem 0..stability_max")
        if not 0 <= cfg.props_start <= cfg.props_max:
            raise ValueError("props_start poza zakresem 0..props_max")
        self.exit_pos = mine_map.exit

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self.random_maps:
            # options={"map_seed": s} oddziela mapę od losowania zawałów (ta sama mapa, nowa gra);
            # bez niego mapa wynika z ziarna epizodu.
            map_seed = (options or {}).get("map_seed")
            map_rng = np.random.default_rng(map_seed) if map_seed is not None else self.np_random
            self._load_map(generate_map(map_rng, self.generator))
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
        load_before = self.load_value
        legal = self._apply(action)
        if not legal:
            reward += cfg.reward_illegal
        if self.last_dug is not None and not cfg.reward_dig_refund:
            reward += cfg.reward_dig_fraction * self._deposit_value[self._deposit_index[self.last_dug]]

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

        if cfg.reward_dig_refund:            # kształtowanie potencjałowe: γ_s·Φ(s') − Φ(s)
            phi_next = 0.0 if terminated else cfg.reward_dig_fraction * self.load_value
            reward += cfg.reward_shaping_gamma * phi_next - cfg.reward_dig_fraction * load_before
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
        self.visited: dict[tuple[int, int], int] = {self.pos: 1}   # pole -> liczba wejść
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

    def action_mask(self) -> np.ndarray:
        """Które akcje są w bieżącym stanie dozwolone (te same warunki co w `_apply`)."""
        cfg = self.config
        mask = np.zeros(self.action_space.n, dtype=bool)
        for action, (dx, dy) in DIRECTIONS.items():
            tx, ty = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= tx < self.width and 0 <= ty < self.height:
                target = self.grid[ty][tx]
                mask[action] = target in DEPOSITS or target in WALKABLE
        mask[PROP] = self.props > 0 and self.stability < cfg.stability_max
        return mask

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
            self.visited[self.pos] = self.visited.get(self.pos, 0) + 1
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
            p = cfg.collapse_floor + cfg.collapse_base \
                * (cfg.stability_max - stability) ** cfg.collapse_power * (1.0 + y / self.height)
        return float(min(max(p, 0.0), 1.0))

    # ---------------------------------------------------------- obserwacja
    def oil_bin(self, oil: int | None = None) -> int:
        """Przedział oleju 0..oil_bins-1 (przy 100 i 4 przedziałach: 0-24, 25-49, 50-74, 75-100)."""
        cfg = self.config
        oil = self.oil if oil is None else oil
        oil = min(max(oil, 0), cfg.oil_start)
        return min(cfg.oil_bins - 1, oil * cfg.oil_bins // max(cfg.oil_start, 1))

    _CELL_CODE = {ROCK: 0, "S": 1, ".": 1, EXIT: 3, COLLAPSING: 4}

    def local_view(self) -> list[int]:
        """Kody pól wokół górnika (wierszami), poza mapą = skała."""
        r = self.view_radius
        px, py = self.pos
        codes = []
        for y in range(py - r, py + r + 1):
            for x in range(px - r, px + r + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    ch = self.grid[y][x]
                    codes.append(2 if ch in DEPOSITS else self._CELL_CODE.get(ch, 0))
                else:
                    codes.append(0)
        return codes

    def _paths(self) -> dict[tuple[int, int], tuple[int, int]]:
        """Najkrótsze drogi od górnika (BFS po polach przechodnich): pole -> (długość,
        kierunek pierwszego kroku 1..4 = góra, dół, lewo, prawo).

        Złoża są celami: można na nie wejść (wydobycie), ale droga przez nie nie prowadzi.
        """
        paths = {self.pos: (0, 0)}
        frontier = [self.pos]
        while frontier:
            nxt = []
            for x, y in frontier:
                d, first = paths[(x, y)]
                for action, (dx, dy) in DIRECTIONS.items():
                    cell = (x + dx, y + dy)
                    if cell in paths or not (0 <= cell[0] < self.width and 0 <= cell[1] < self.height):
                        continue
                    ch = self.grid[cell[1]][cell[0]]
                    if ch in DEPOSITS or ch in WALKABLE:
                        paths[cell] = (d + 1, first or action + 1)
                    if ch in WALKABLE:
                        nxt.append(cell)
            frontier = nxt
        return paths

    def _direction(self, target: tuple[int, int], first_step: int) -> int:
        if self.hint_kind == "path":
            return first_step
        dx, dy = target[0] - self.pos[0], target[1] - self.pos[1]
        return COMPASS.get((int(np.sign(dx)), int(np.sign(dy))), 0)

    def hint(self) -> list[int]:
        """Podpowiedź narratora: (kierunek, odległość, rodzaj) najbliższego osiągalnego
        niewydobytego złoża i (kierunek, odległość) do wyjścia; 0 = brak celu."""
        paths = self._paths()
        reachable = [(paths[(x, y)][0], i) for i, (x, y, _) in enumerate(self.deposits)
                     if not self.mined[i] and (x, y) in paths]
        if reachable:
            d, i = min(reachable)
            x, y, ch = self.deposits[i]
            deposit = [self._direction((x, y), paths[(x, y)][1]), int(np.searchsorted(DIST_BINS, d)),
                       DEPOSITS.index(ch) + 1]
        else:
            deposit = [0, 0, 0]
        if self.exit_pos in paths:
            d, first = paths[self.exit_pos]
            exit_hint = [self._direction(self.exit_pos, first), int(np.searchsorted(DIST_BINS, d))]
        else:
            exit_hint = [0, 0]
        return deposit + exit_hint

    def neighbours(self) -> list[int]:
        """Kody 4 sąsiednich pól (góra, dół, lewo, prawo) jak w `local_view`."""
        codes = []
        for dx, dy in DIRECTIONS.values():
            x, y = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                ch = self.grid[y][x]
                codes.append(2 if ch in DEPOSITS else self._CELL_CODE.get(ch, 0))
            else:
                codes.append(0)
        return codes

    def _obs(self) -> np.ndarray:
        if self.obs_mode == "hint":
            load_bin = int(np.searchsorted(LOAD_BINS, self.load_value))
            values = [self.stability, self.oil_bin(), self.props, load_bin, *self.neighbours(), *self.hint()]
            if self.hint_memory:
                values += [min(self.visited.get((self.pos[0] + dx, self.pos[1] + dy), 0), self.hint_memory)
                           for dx, dy in DIRECTIONS.values()]
            return np.array(values, dtype=np.int64)
        values = [self.pos[0], self.pos[1], self.stability, self.oil_bin(), *self.mined, self.props]
        if self.obs_mode == "local":
            values += self.local_view()
        return np.array(values, dtype=np.int64)

    def state_key(self) -> tuple[int, ...]:
        """Pełny stan (x, y, sigma, o, m, n) niezależnie od trybu obserwacji."""
        return (*self.pos, self.stability, self.oil_bin(), *(int(m) for m in self.mined), self.props)

    def decode(self, obs) -> State:
        """Rozpakowanie obserwacji na nazwane pola (tryby "state" i "local")."""
        if self.obs_mode == "hint":
            raise ValueError("Obserwacja z podpowiedzią nie zawiera pozycji ani maski złóż")
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
        total = 1
        for radix in self.observation_space.nvec:
            total *= int(radix)
        return total

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
        """Wektor cech w [0, 1] (np. dla DQN), długość jak obserwacja."""
        nvec = self.observation_space.nvec
        values = np.asarray(obs, dtype=np.float32).ravel()
        return values / np.maximum(nvec - 1, 1).astype(np.float32)

    @property
    def vector_size(self) -> int:
        return len(self.observation_space.nvec)

    # ------------------------------------------------------------ pomocnicze
    def to_text(self) -> str:
        rows = ["".join(r) for r in self.grid]
        x, y = self.pos
        rows[y] = rows[y][:x] + "@" + rows[y][x + 1:]
        hud = (f"krok {self.steps}  stabilnosc {self.stability}/{self.config.stability_max}  "
               f"olej {self.oil}  ladunek {self.load_value}  stemple {self.props}")
        return "\n".join(rows + [hud])
