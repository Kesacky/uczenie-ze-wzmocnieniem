"""Wczytywanie map tekstowych z `maps/*.txt`.

Format pliku:

    ; nazwa = Krucha kopalnia          <- opcjonalny nagłówek: `; klucz = wartość`
    ; stability_start = 3              <- klucze inne niż `nazwa`/`opis` nadpisują GameConfig
    ###E####                           <- wiersz 0 to powierzchnia z wyjściem E
    #..S.c.#
    ...

Znaki: `#` lita skała (nieprzechodnia), `.` korytarz, `S` start, `E` wyjście,
`c` węgiel, `s` srebro, `g` złoto, `k` kryształ (wejście na złoże = wydobycie),
`X` korytarz, który po wydobyciu N złóż zamienia się w skałę.
Wszystkie wiersze muszą mieć tę samą długość.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miner_rl.config import parse_value

ROCK = "#"
CORRIDOR = "."
START = "S"
EXIT = "E"
COLLAPSING = "X"
DEPOSITS = ("c", "s", "g", "k")
FILE_CHARS = set("#.SEX") | set(DEPOSITS)
WALKABLE = {CORRIDOR, START, EXIT, COLLAPSING}   # pola, na które wchodzi się bez wydobycia

INFO_KEYS = {"wygenerowano"}           # klucze nagłówka, które nie zmieniają GameConfig
MAPS_DIR = Path(__file__).resolve().parents[2] / "maps"


@dataclass(frozen=True)
class MineMap:
    name: str
    rows: tuple[str, ...]
    overrides: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def height(self) -> int:
        return len(self.rows)

    def cells(self, chars: set[str] | tuple[str, ...]) -> list[tuple[int, int]]:
        """Pozycje (x, y) pól o danych znakach, w kolejności wierszami."""
        return [
            (x, y)
            for y, row in enumerate(self.rows)
            for x, ch in enumerate(row)
            if ch in chars
        ]

    @property
    def start(self) -> tuple[int, int]:
        return self.cells({START})[0]

    @property
    def exit(self) -> tuple[int, int]:
        return self.cells({EXIT})[0]

    @property
    def deposits(self) -> list[tuple[int, int, str]]:
        """Złoża w ustalonej kolejności; indeks = bit w masce obserwacji."""
        return [(x, y, self.rows[y][x]) for x, y in self.cells(DEPOSITS)]

    @property
    def collapse_cells(self) -> list[tuple[int, int]]:
        return self.cells({COLLAPSING})


def parse_map(text: str, name: str = "mapa") -> MineMap:
    overrides: dict[str, Any] = {}
    description = ""
    rows: list[str] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r\n")
        if line.startswith(";"):
            key, sep, value = line[1:].partition("=")
            if not sep:
                continue
            key = key.strip()
            if key == "nazwa":
                name = value.strip()
            elif key == "opis":
                description = value.strip()
            elif key in INFO_KEYS:
                continue                      # informacja dla czytelnika, nie konfiguracja
            else:
                overrides[key] = parse_value(value)
            continue
        line = line.rstrip()
        if not line:
            continue
        bad = set(line) - FILE_CHARS
        if bad:
            raise ValueError(f"{name}: niedozwolone znaki {sorted(bad)} w linii {line_no}")
        rows.append(line)
    _validate(rows, name)
    return MineMap(name=name, rows=tuple(rows), overrides=overrides, description=description)


def load_map(path: str | Path) -> MineMap:
    """Wczytaj mapę z pliku; sama nazwa (np. `m1_stabilna`) szuka w katalogu maps/."""
    path = Path(path)
    if not path.suffix and not path.exists():
        path = MAPS_DIR / f"{path.name}.txt"
    return parse_map(path.read_text(encoding="utf-8"), name=path.stem)


def list_maps() -> list[Path]:
    return sorted(MAPS_DIR.glob("*.txt"))


def _validate(rows: list[str], name: str) -> None:
    if not rows:
        raise ValueError(f"{name}: pusta mapa")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{name}: wiersze mają różne długości {sorted(widths)}")
    flat = "".join(rows)
    if flat.count(START) != 1:
        raise ValueError(f"{name}: wymagany dokładnie jeden start S")
    if flat.count(EXIT) != 1:
        raise ValueError(f"{name}: wymagane dokładnie jedno wyjście E")
    if EXIT not in rows[0]:
        raise ValueError(f"{name}: wyjście E musi leżeć w wierszu 0 (powierzchnia)")


def reachable(rows: list[str] | tuple[str, ...], start: tuple[int, int],
              walkable: set[str] | frozenset[str] = frozenset(WALKABLE | set(DEPOSITS)),
              ) -> set[tuple[int, int]]:
    """Pola osiągalne ze `start` (BFS, 4-sąsiedztwo). Domyślnie złoża są przechodnie,
    bo wejście na złoże to jego wydobycie."""
    height, width = len(rows), len(rows[0])
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in seen \
                    and rows[ny][nx] in walkable:
                seen.add((nx, ny))
                queue.append((nx, ny))
    return seen
