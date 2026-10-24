"""Wszystkie stałe gry i nagród w jednym miejscu.

Wartości domyślne odpowiadają formalizacji MDP z raportu (`report/main.tex`,
sekcja „Formalizacja RL/MDP”, decyzje z 21.10.2026). Każda liczba, od której
zależy dynamika lub nagroda, jest polem `GameConfig`, więc jej zmiana nie
wymaga zmian w kodzie środowiska. Konfigurację można wczytać z pliku JSON lub
YAML (YAML tylko z zainstalowanym PyYAML) albo nadpisać w kodzie:

    cfg = GameConfig().replace(collapse_base=0.01)
    cfg = GameConfig.from_file("configs/kara_50.json")

Mapa może nadpisać część pól w nagłówku (linie `; klucz = wartość`), np.
krucha kopalnia zaczyna z niższą stabilnością. Pierwszeństwo:
domyślne < nagłówek mapy < jawne nadpisanie przekazane do `MineEnv`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GameConfig:
    # --- Złoża: znak mapy -> wartość (głębsze są cenniejsze) ---
    deposit_values: dict[str, int] = field(
        default_factory=lambda: {"c": 3, "s": 9, "g": 18, "k": 30}   # 21.10.2026: ×3 dla map 16×16
    )
    max_deposits: int = 5                # K: górna granica liczby złóż na mapie

    # --- Stabilność stropu (globalna dla całej kopalni), sigma w 0..stability_max ---
    stability_max: int = 5
    stability_start: int = 5
    dig_stability_cost: int = 1          # wydobycie złoża obniża stabilność o tyle
    prop_stability_gain: int = 2         # `podepnij` podnosi stabilność o tyle
    props_start: int = 2                 # n: liczba stempli na starcie
    props_max: int = 2                   # górna granica n (rozmiar przestrzeni stanów)

    # --- Zawał: p = collapse_floor + collapse_base * (stability_max - sigma)^collapse_power * (1 + y / H) ---
    collapse_floor: float = 0.001        # ryzyko w każdym kroku nawet przy pełnej stabilności
    collapse_base: float = 0.003
    collapse_power: float = 2.0          # >1: pierwsze wydobycia mało groźne, każde kolejne coraz bardziej
    collapse_prob: float | None = None   # jeśli ustawione, wymusza stałe p (np. 1.0 w testach)

    # --- Olej w lampie ---
    oil_start: int = 100
    oil_per_step: int = 1
    oil_bins: int = 4                    # o: olej w przedziałach 0..oil_bins-1

    # --- Zdarzenie: po wydobyciu N złóż pola `X` zamieniają się w skałę ---
    corridor_event_after: int = 3

    # --- Nagrody (składniki się sumują) ---
    reward_step: float = -0.1            # za każdy krok
    reward_illegal: float = -2.0         # dodatkowo za akcję niedozwoloną (stan bez zmian); do 24.10.2026: -0.5
    reward_collapse: float = -10.0       # zawał, ładunek przepada; do 24.10.2026: -20
    reward_out_of_oil: float = -20.0     # wyczerpanie oleju, ładunek przepada
    reward_exit_multiplier: float = 1.0  # nagroda przy wyjściu = mnożnik * wartość ładunku
    # Nagroda za wydobycie: ułamek wartości złoża wypłacany od razu (od 24.10.2026 domyślnie 0,3,
    # wcześniej 0). Bez `reward_dig_refund` jest to premia: ładunek i tak jest wypłacany przy
    # wyjściu. Z `reward_dig_refund` jest to kształtowanie potencjałowe (Ng i in. 1999) z
    # potencjałem Φ(s) = ułamek · ładunek: w każdym kroku nagroda dostaje γ_s·Φ(s') − Φ(s),
    # gdzie Φ = 0 w stanie terminalnym, a γ_s = `reward_shaping_gamma` (powinno być równe γ
    # agenta, wtedy polityka optymalna się nie zmienia).
    reward_dig_fraction: float = 0.3
    reward_dig_refund: bool = False
    reward_shaping_gamma: float = 1.0

    # --- Epizod ---
    step_limit: int = 300                # techniczny limit kroków (truncated)

    def replace(self, **changes: Any) -> "GameConfig":
        """Kopia z nadpisanymi polami (nieznane klucze zgłaszają błąd)."""
        return replace(self, **_coerce(changes))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameConfig":
        return cls().replace(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> "GameConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as error:  # pragma: no cover - zależy od środowiska
                raise ImportError("Do plików YAML potrzebny jest pakiet PyYAML.") from error
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)


_FIELD_NAMES = {f.name for f in fields(GameConfig)}


def _coerce(changes: dict[str, Any]) -> dict[str, Any]:
    unknown = set(changes) - _FIELD_NAMES
    if unknown:
        raise KeyError(f"Nieznane pola konfiguracji: {sorted(unknown)}")
    out = dict(changes)
    if "deposit_values" in out:
        out["deposit_values"] = {str(k): int(v) for k, v in out["deposit_values"].items()}
    return out


def parse_value(raw: str) -> Any:
    """Zamiana tekstu z nagłówka mapy na wartość (liczba, bool, None, JSON)."""
    raw = raw.strip()
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
