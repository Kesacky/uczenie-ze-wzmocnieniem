"""Automatyczne generowanie kopalni (przekrój pionowy).

Mapa ma wyglądać jak kopalnia, a nie jak labirynt: szyb, poziomy wydobywcze,
łączniki i krótkie przodki, a dookoła lita skała. Wszystkie losowania pochodzą
z przekazanego `np.random.Generator`, więc mapa jest w pełni wyznaczona przez ziarno.

1. **Wyjście i nadszybie.** Wyjście `E` w wierszu 0 nad losową kolumną, start `S`
   tuż pod nim. W wierszu 1 krótki chodnik przy szybie (nadszybie).
2. **Poziomy.** Wybierane są wiersze poziomów wydobywczych co 2–3 wiersze
   w dół aż do dna. Każdy poziom to poziomy chodnik o losowej długości
   w lewo i w prawo od miejsca, w którym dochodzi do niego szyb.
3. **Szyby.** Szyb główny schodzi pionowo od startu do jednego z poziomów;
   głębsze poziomy łączą się szybikami (łącznikami) w losowej kolumnie,
   wspólnej dla obu sąsiednich chodników (chodnik wydłuża się w razie potrzeby).
4. **Dodatkowe łączniki.** Z prawdopodobieństwem `extra_winze_prob` dwa
   sąsiednie poziomy dostają drugi łącznik – powstaje droga alternatywna.
5. **Przodki i komory.** Z chodników odchodzą krótkie ślepe wyrobiska w górę
   lub w dół (1–2 pola), a w kilku miejscach chodnik poszerza się w komorę.
   Wyrobiska nie mogą stykać się bokiem z innymi, żeby nie powstawała siatka.
6. **Złoża.** Kandydatami są ślepe końce wyrobisk (końce chodników i przodków),
   co najmniej `min_start_distance` pól od startu. Zakres głębokości dzielony
   jest na pasy (domyślnie K równych pasów); w każdym pasie wybierany jest jeden
   kandydat, przy czym głębsze złoża muszą leżeć dalej od startu. Rodzaj złoża
   zależy od głębokości: płytko węgiel, potem srebro, złoto, na dnie kryształ.
7. **Korytarz X (opcjonalnie).** Odcinek 2–3 pól na najkrótszej drodze do
   najgłębszego złoża, po którego zawaleniu wyjście i wszystkie złoża pozostają
   osiągalne, a droga wydłuża się co najmniej o `min_detour` (zwykle wtedy
   trzeba wracać łącznikiem zamiast szybem).
8. **Walidacja.** Wyjście i wszystkie złoża muszą być osiągalne ze startu (także
   po zawale X); przy niepowodzeniu losowanie jest powtarzane.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from miner_rl.env.maps import COLLAPSING, CORRIDOR, DEPOSITS, EXIT, ROCK, START, MineMap

DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))


@dataclass(frozen=True)
class GeneratorConfig:
    width: int = 12
    height: int = 11
    n_deposits: int = 5
    level_gap: tuple[int, int] = (2, 3)      # odstęp między poziomami (w wierszach)
    extra_winze_prob: float = 0.5            # szansa drugiego łącznika między poziomami
    stub_prob: float = 0.35                  # szansa przodka przy danym polu chodnika (na poziom maks. 2)
    chamber_prob: float = 0.12               # szansa poszerzenia chodnika w komorę
    min_start_distance: int = 4              # złoża nie bliżej startu (w polach)
    depth_bands: tuple[tuple[float, float], ...] | None = None   # własne pasy głębokości 0..1
    type_jitter: float = 0.08                # losowe przesunięcie głębokości przy wyborze rodzaju
    collapse_prob: float = 0.5               # szansa, że mapa ma korytarz X
    min_detour: int = 3                      # minimalne wydłużenie drogi po zawale korytarza
    overrides: dict = field(default_factory=dict)   # nadpisania GameConfig zapisywane w mapie
    max_tries: int = 400


def deposit_type(depth_fraction: float) -> str:
    """Rodzaj złoża wg względnej głębokości 0 (powierzchnia) .. 1 (dno)."""
    if depth_fraction < 0.35:
        return "c"
    if depth_fraction < 0.55:
        return "s"
    if depth_fraction < 0.78:
        return "g"
    return "k"


def generate_map(rng: np.random.Generator, gen: GeneratorConfig | None = None,
                 name: str = "losowa") -> MineMap:
    gen = gen or GeneratorConfig()
    if gen.width < 6 or gen.height < 6:
        raise ValueError("Mapa musi mieć co najmniej 6×6 pól")
    for _ in range(gen.max_tries):
        rows = _try_generate(rng, gen)
        if rows is not None:
            return MineMap(name=name, rows=tuple("".join(r) for r in rows),
                           overrides=dict(gen.overrides),
                           description="Mapa wygenerowana automatycznie")
    raise RuntimeError("Nie udało się wygenerować poprawnej mapy; zmień parametry generatora")


# ---------------------------------------------------------------- etapy
def _try_generate(rng: np.random.Generator, gen: GeneratorConfig):
    w, h = gen.width, gen.height
    grid = [[ROCK] * w for _ in range(h)]
    lo_x, hi_x = 1, w - 2                       # kolumny wnętrza
    bottom = h - 2                              # najniższy wiersz wnętrza

    def carve(x, y):
        if lo_x <= x <= hi_x and 1 <= y <= bottom:
            grid[y][x] = CORRIDOR

    # 1. wyjście, start i nadszybie
    sx = int(rng.integers(2, w - 2))
    carve(sx, 1)
    for _ in range(int(rng.integers(0, 3))):
        carve(sx - 1 - _, 1)
    for _ in range(int(rng.integers(0, 3))):
        carve(sx + 1 + _, 1)

    # 2. poziomy
    levels = []
    y = 1
    while True:
        y += int(rng.integers(gen.level_gap[0], gen.level_gap[1] + 1))
        if y > bottom:
            break
        levels.append(y)
    if len(levels) < 2:
        return None
    if bottom - levels[-1] >= 1 and rng.random() < 0.5:
        levels[-1] = bottom                     # najgłębszy poziom czasem na samym dnie

    # 3. szyb główny i łączniki
    spans: dict[int, list[int]] = {}

    def gallery(level_y, anchor_x):
        left = int(rng.integers(1, max(2, anchor_x - lo_x + 1)))
        right = int(rng.integers(1, max(2, hi_x - anchor_x + 1)))
        a, b = max(lo_x, anchor_x - left), min(hi_x, anchor_x + right)
        spans[level_y] = [a, b]
        for x in range(a, b + 1):
            carve(x, level_y)

    main_depth = int(rng.integers(0, len(levels)))      # do którego poziomu sięga szyb główny
    for yy in range(1, levels[main_depth] + 1):
        carve(sx, yy)
    for i, ly in enumerate(levels):
        gallery(ly, sx if i <= main_depth else anchor)
        if i < main_depth:
            continue
        if i + 1 < len(levels):
            a, b = spans[ly]
            anchor = int(rng.integers(a, b + 1))
            if anchor == sx and b > a:                   # łącznik obok szybu, nie jego przedłużenie
                anchor = a if sx != a else b
            for yy in range(ly, levels[i + 1] + 1):
                carve(anchor, yy)
    # chodniki płytszych poziomów przechodzą przez szyb – szyb już je łączy

    # 4. dodatkowe łączniki
    for upper, lower in zip(levels, levels[1:]):
        if rng.random() < gen.extra_winze_prob:
            a = max(spans[upper][0], spans[lower][0])
            b = min(spans[upper][1], spans[lower][1])
            options = [x for x in range(a, b + 1)
                       if all(grid[yy][x] == ROCK for yy in range(upper + 1, lower))
                       and all(grid[yy][x + d] == ROCK for yy in range(upper + 1, lower) for d in (-1, 1)
                               if lo_x <= x + d <= hi_x)]
            if options:
                x = options[int(rng.integers(len(options)))]
                for yy in range(upper + 1, lower):
                    carve(x, yy)

    # 5. przodki i komory
    for ly in levels:
        a, b = spans[ly]
        made = 0
        for x in rng.permutation(np.arange(a, b + 1)):
            if made >= 2 or rng.random() > gen.stub_prob:
                continue
            dy = -1 if rng.random() < 0.5 else 1
            length = int(rng.integers(1, 3))
            cells = [(int(x), ly + dy * k) for k in range(1, length + 1)]
            prevs = [(int(x), ly)] + cells[:-1]
            if all(_free_for_stub(grid, c, pr, lo_x, hi_x, bottom) for c, pr in zip(cells, prevs)):
                for c in cells:
                    carve(*c)
                made += 1
        for x in range(a, b + 1):
            if rng.random() < gen.chamber_prob:
                cy = ly - 1
                if cy >= 2 and grid[cy - 1][x] == ROCK:
                    carve(x, cy)

    grid[0][sx] = EXIT
    grid[1][sx] = START
    start = (sx, 1)

    # 6. złoża na ślepych końcach, rozłożone po głębokości
    dist = _bfs(grid, start)
    dead_ends = [p for p in dist if grid[p[1]][p[0]] == CORRIDOR and _degree(grid, p) == 1
                 and dist[p] >= gen.min_start_distance]
    if len(dead_ends) < gen.n_deposits:
        return None
    bands = gen.depth_bands or tuple((i / gen.n_deposits, (i + 1) / gen.n_deposits)
                                     for i in range(gen.n_deposits))
    if len(bands) != gen.n_deposits:
        raise ValueError("Liczba pasów głębokości musi równać się n_deposits")
    span = max(bottom - 1, 1)
    chosen: list[tuple[int, int]] = []
    for lo, hi in sorted(bands):
        band = [p for p in dead_ends if p not in chosen
                and lo - 1e-9 <= (p[1] - 1) / span <= hi + 1e-9]
        if not band:
            band = [p for p in dead_ends if p not in chosen and (p[1] - 1) / span >= lo - 0.25]
        if not band:
            return None
        band.sort()
        chosen.append(band[int(rng.integers(len(band)))])
    for x, y in chosen:
        f = (y - 1) / span + rng.uniform(-gen.type_jitter, gen.type_jitter)
        grid[y][x] = deposit_type(float(np.clip(f, 0.0, 1.0)))

    # 7. korytarz X
    if rng.random() < gen.collapse_prob:
        segment = _collapse_segment(grid, start, chosen, rng, gen.min_detour)
        if segment is None:
            return None
        for x, y in segment:
            grid[y][x] = COLLAPSING

    # 8. walidacja
    if not _valid(grid, start):
        return None
    return grid


def _free_for_stub(grid, cell, previous, lo_x, hi_x, bottom) -> bool:
    """Pole przodka: skała we wnętrzu, której jedynym sąsiadem-wyrobiskiem jest `previous`."""
    x, y = cell
    if not (lo_x <= x <= hi_x and 2 <= y <= bottom) or grid[y][x] != ROCK:
        return False
    for dx, dy in DIRS:
        n = (x + dx, y + dy)
        if n != previous and 0 <= n[1] < len(grid) and 0 <= n[0] < len(grid[0]) \
                and grid[n[1]][n[0]] != ROCK:
            return False
    return True


def _degree(grid, p) -> int:
    h, w = len(grid), len(grid[0])
    x, y = p
    return sum(1 for dx, dy in DIRS
               if 0 <= x + dx < w and 0 <= y + dy < h and grid[y + dy][x + dx] != ROCK)


def _bfs(grid, start, blocked=frozenset()):
    """Odległości w polach; złoża są przechodnie (wejście = wydobycie)."""
    h, w = len(grid), len(grid[0])
    dist = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in dist and (nx, ny) not in blocked \
                    and grid[ny][nx] != ROCK:
                dist[(nx, ny)] = dist[(x, y)] + 1
                queue.append((nx, ny))
    return dist


def _path(grid, start, goal):
    h, w = len(grid), len(grid[0])
    prev = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        x, y = cur
        for dx, dy in DIRS:
            nxt = (x + dx, y + dy)
            if 0 <= nxt[0] < w and 0 <= nxt[1] < h and nxt not in prev and grid[nxt[1]][nxt[0]] != ROCK:
                prev[nxt] = cur
                queue.append(nxt)
    if goal not in prev:
        return None
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def _collapse_segment(grid, start, deposits, rng, min_detour):
    target = max(deposits, key=lambda p: p[1])
    path = _path(grid, start, target)
    if path is None:
        return None
    base = len(path) - 1
    exit_pos = (start[0], 0)
    candidates = []
    for length in (3, 2):
        for i in range(1, len(path) - length):
            seg = path[i:i + length]
            if any(grid[y][x] != CORRIDOR for x, y in seg):
                continue
            dist = _bfs(grid, start, blocked=frozenset(seg))
            if exit_pos not in dist or any(d not in dist for d in deposits):
                continue
            if dist[target] - base >= min_detour:
                candidates.append(tuple(seg))
        if candidates:
            break
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _valid(grid, start) -> bool:
    dist = _bfs(grid, start)
    h, w = len(grid), len(grid[0])
    exit_pos = (start[0], 0)
    deposits = [(x, y) for y in range(h) for x in range(w) if grid[y][x] in DEPOSITS]
    if exit_pos not in dist or any(d not in dist for d in deposits):
        return False
    xs = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == COLLAPSING]
    if xs:
        after = _bfs(grid, start, blocked=frozenset(xs))
        if exit_pos not in after or any(d not in after for d in deposits):
            return False
    return True


def map_to_text(mine_map: MineMap, header: dict | None = None) -> str:
    """Tekst pliku mapy (format maps/*.txt) z nagłówkiem."""
    lines = []
    for key, value in (header or {}).items():
        lines.append(f"; {key} = {value}")
    for key, value in mine_map.overrides.items():
        lines.append(f"; {key} = {value}")
    lines.extend(mine_map.rows)
    return "\n".join(lines) + "\n"
