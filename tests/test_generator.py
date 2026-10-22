"""Testy generatora map."""

import numpy as np
import pytest

from miner_rl.env import GeneratorConfig, MineEnv, generate_map
from miner_rl.env.generator import deposit_type
from miner_rl.env.maps import COLLAPSING, ROCK, reachable


@pytest.mark.parametrize("seed", range(40))
def test_generated_maps_are_valid(seed):
    m = generate_map(np.random.default_rng(seed))
    assert (m.width, m.height) == (12, 11) and len(m.deposits) == 5
    assert m.exit[1] == 0 and m.start == (m.exit[0], 1)
    assert all(ch == ROCK for ch in m.rows[-1])                      # dolna ramka
    assert all(r[0] == ROCK and r[-1] == ROCK for r in m.rows)        # boczne ramki
    reach = reachable(m.rows, m.start)
    assert m.exit in reach and all((x, y) in reach for x, y, _ in m.deposits)
    if m.collapse_cells:                                              # droga alternatywna
        rows = [r.replace(COLLAPSING, ROCK) for r in m.rows]
        after = reachable(rows, m.start)
        assert m.exit in after and all((x, y) in after for x, y, _ in m.deposits)
    MineEnv(m)                                                        # przechodzi walidację środowiska


def test_generator_is_deterministic():
    a = generate_map(np.random.default_rng(123))
    b = generate_map(np.random.default_rng(123))
    c = generate_map(np.random.default_rng(124))
    assert a.rows == b.rows != c.rows


def test_deposit_types_follow_depth():
    assert [deposit_type(f) for f in (0.0, 0.4, 0.6, 0.9)] == ["c", "s", "g", "k"]
    deep = GeneratorConfig(depth_bands=((.85, 1),) * 5, collapse_prob=0.0, type_jitter=0.0)
    for seed in range(5):
        m = generate_map(np.random.default_rng(seed), deep)
        assert all(ch in "gk" for *_, ch in m.deposits)       # na małej mapie pas może się poszerzyć
        assert sum(ch == "k" for *_, ch in m.deposits) >= 2


def test_collapse_corridor_and_custom_size():
    gen = GeneratorConfig(width=16, height=14, n_deposits=4, collapse_prob=1.0)
    for seed in range(10):
        m = generate_map(np.random.default_rng(seed), gen)
        assert (m.width, m.height) == (16, 14) and len(m.deposits) == 4
        assert 2 <= len(m.collapse_cells) <= 3


def test_fixed_maps_match_generator_script():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gen_maps", root / "scripts" / "gen_maps.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, (theme, seed) in module.FIXED.items():
        _, text = module.make(theme, seed, name)
        assert (root / "maps" / f"{name}.txt").read_text(encoding="utf-8") == text


@pytest.mark.parametrize("seed", range(40))
def test_generated_maps_look_like_mines(seed):
    """Kopalnia, nie labirynt: dużo litej skały, złoża daleko od startu, rozłożone po głębokości."""
    from miner_rl.env.generator import _bfs
    m = generate_map(np.random.default_rng(seed))
    inner = [ch for row in m.rows[1:-1] for ch in row[1:-1]]
    assert inner.count(ROCK) / len(inner) > 0.45
    dist = _bfs([list(r) for r in m.rows], m.start)
    assert all(dist[(x, y)] >= 4 for x, y, _ in m.deposits)
    depths = sorted(y for _, y, _ in m.deposits)
    assert depths[-1] - depths[0] >= 3


def test_maps_differ_between_seeds():
    layouts = {generate_map(np.random.default_rng(s)).rows for s in range(30)}
    assert len(layouts) == 30
