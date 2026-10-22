"""Testy logiki MineEnv (bez grafiki), zgodne z formalizacją MDP z raportu."""

from __future__ import annotations

import json

import numpy as np
import pytest

from miner_rl.config import GameConfig
from miner_rl.env import MineEnv, list_maps, load_map, parse_map
from miner_rl.env.maps import reachable
from miner_rl.env.mine_env import DOWN, LEFT, PROP, RIGHT, UP

# Mała mapa testowa: E w (2,0), start S w (2,1), węgiel c w (1,1), złoto g w (1,3),
# pole X w (2,2) zamienia się w skałę po zdarzeniu.
TINY = """\
##E##
#cS.#
#.X.#
#g..#
#####
"""
# Mapa bez obramowania: z S w (0,1) ruch w lewo wychodzi poza mapę.
EDGE = """\
E.c
S..
"""

SAFE = dict(collapse_prob=0.0)   # wyłącza losowe zawały tam, gdzie test ich nie dotyczy


def make(text=TINY, **overrides):
    env = MineEnv(parse_map(text, "test"), **overrides)
    obs, info = env.reset(seed=0)
    return env, obs, info


def snapshot(env):
    return (env.pos, env.stability, env.props, env.load_value,
            tuple(env.mined), tuple(map(tuple, env.grid)))


# ------------------------------------------------------------------ reset
def test_reset_returns_state_tuple_from_report():
    env, obs, info = make(**SAFE)
    assert env.observation_space.contains(obs)
    # s = (x, y, sigma, o, m_1..m_K, n)
    assert obs.tolist() == [2, 1, 5, 3, 0, 0, 2]
    st = env.decode(obs)
    assert (st.x, st.y, st.stability, st.oil, st.mined, st.props) == (2, 1, 5, 3, (0, 0), 2)
    assert info["reason"] is None and info["steps"] == 0
    assert env.action_space.n == 5


def run_trajectory(seed, actions, **overrides):
    env = MineEnv("m3_gleboka", **overrides)
    obs, _ = env.reset(seed=seed)
    trace = [env.obs_to_key(obs)]
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        trace.append((env.obs_to_key(obs), r, term, trunc, info["reason"]))
        if term or trunc:
            break
    return trace


def test_same_seed_gives_identical_trajectory():
    actions = np.random.default_rng(123).integers(0, 5, size=300).tolist()
    assert run_trajectory(7, actions) == run_trajectory(7, actions)


def test_different_seeds_change_random_collapses():
    actions = [LEFT, RIGHT] * 50          # chodzenie po korytarzu przy p = 0.5
    lengths = {len(run_trajectory(s, actions, collapse_prob=0.5)) for s in range(10)}
    assert len(lengths) > 1


def test_reset_restores_initial_state():
    env, obs0, _ = make(**SAFE)
    env.step(LEFT)                        # wydobycie c
    obs1, _ = env.reset(seed=0)
    assert env.obs_to_key(obs1) == env.obs_to_key(obs0)
    assert env.grid[1][1] == "c" and env.steps == 0 and env.load_value == 0


# --------------------------------------------------------- niedozwolone
def test_move_into_rock_is_illegal_and_changes_nothing():
    env, _, _ = make(**SAFE)
    env.step(RIGHT)                       # (3,1)
    before = snapshot(env)
    _, r, term, trunc, info = env.step(RIGHT)   # (4,1) to skała
    assert info["illegal"] and not term and not trunc
    assert r == pytest.approx(env.config.reward_step + env.config.reward_illegal)
    assert snapshot(env) == before


def test_move_outside_map_is_illegal():
    env, _, _ = make(EDGE, **SAFE)
    before = snapshot(env)
    _, r, _, _, info = env.step(LEFT)
    assert info["illegal"] and snapshot(env) == before
    assert r == pytest.approx(env.config.reward_step + env.config.reward_illegal)


def test_time_still_passes_on_illegal_action():
    env, _, _ = make(**SAFE)
    env.step(RIGHT)
    env.step(RIGHT)
    assert env.steps == 2 and env.oil == env.config.oil_start - 2


# ------------------------------------------------------- wydobycie
def test_entering_deposit_mines_it():
    env, _, _ = make(**SAFE)
    obs, r, _, _, info = env.step(LEFT)   # c w (1,1)
    cfg = env.config
    value = cfg.deposit_values["c"]
    st = env.decode(obs)
    assert not info["illegal"] and (st.x, st.y) == (1, 1)
    assert env.load_value == value == info["collected_value"] == env.load_from_mask(st.mined)
    assert st.stability == cfg.stability_start - cfg.dig_stability_cost
    assert st.mined == (1, 0)
    assert env.grid[1][1] == "." and env.last_dug == (1, 1)
    assert r == pytest.approx(cfg.reward_step)


def test_mined_cell_is_plain_corridor_afterwards():
    env, _, _ = make(**SAFE)
    env.step(LEFT)
    env.step(RIGHT)                       # z powrotem na S
    _, _, _, _, info = env.step(LEFT)     # ponowne wejście: zwykły ruch
    assert not info["illegal"] and env.load_value == env.config.deposit_values["c"]
    assert env.stability == env.config.stability_start - 1 and env.last_dug is None


def test_stability_never_below_zero():
    env, _, _ = make(**SAFE, stability_start=0)
    env.step(LEFT)
    assert env.stability == 0


# ------------------------------------------------------------ stemple
def test_prop_without_props_is_illegal():
    env, _, _ = make(**SAFE, props_start=0, stability_start=2)
    before = snapshot(env)
    _, r, _, _, info = env.step(PROP)
    assert info["illegal"] and snapshot(env) == before
    assert r == pytest.approx(env.config.reward_step + env.config.reward_illegal)


def test_prop_at_full_stability_is_illegal():
    env, _, _ = make(**SAFE)
    _, _, _, _, info = env.step(PROP)
    assert info["illegal"] and env.props == env.config.props_start


def test_prop_raises_stability_and_uses_one_prop():
    env, _, _ = make(**SAFE, stability_start=2, props_start=2)
    obs, _, _, _, _ = env.step(PROP)
    assert env.stability == 4 and env.props == 1 and env.prop_cells == [(2, 1)]
    assert env.decode(obs).props == 1
    env.step(PROP)                        # 4 + 2 -> obcięte do 5
    assert env.stability == env.config.stability_max and env.props == 0


# -------------------------------------------------------------- zawał
def test_forced_collapse_ends_episode_and_loses_load():
    env, _, _ = make(collapse_prob=0.0)
    env.step(LEFT)                        # ładunek 1
    env.config = env.config.replace(collapse_prob=1.0)
    _, r, term, trunc, info = env.step(DOWN)
    cfg = env.config
    assert term and not trunc
    assert info["reason"] == "collapse" and info["collapsed"]
    assert info["lost_value"] == cfg.deposit_values["c"]
    assert r == pytest.approx(cfg.reward_step + cfg.reward_collapse)
    with pytest.raises(RuntimeError):
        env.step(UP)


def test_collapse_probability_formula():
    env, _, _ = make()
    cfg = env.config
    assert (cfg.collapse_floor, cfg.collapse_base, cfg.collapse_power) == (0.001, 0.003, 2.0)
    p = env.collapse_probability(stability=2, y=3)
    assert p == pytest.approx(0.001 + 0.003 * (5 - 2) ** 2 * (1 + 3 / env.height))
    # przy pełnej stabilności zostaje tylko ryzyko podstawowe 0,1% na krok
    assert env.collapse_probability(stability=cfg.stability_max, y=3) == pytest.approx(0.001)
    # po jednym wydobyciu ryzyko rośnie łagodnie (kwadrat: 1, 4, 9, 16, 25)
    assert env.collapse_probability(stability=4, y=3) < 0.006
    env.config = cfg.replace(collapse_base=10.0)
    assert env.collapse_probability(stability=0, y=3) == 1.0   # obcięcie do [0, 1]


def test_full_stability_never_collapses_without_floor():
    env = MineEnv("m1_stabilna", step_limit=200, oil_start=1000, collapse_floor=0.0)
    env.reset(seed=1)
    for i in range(200):
        _, _, term, trunc, info = env.step(RIGHT if i % 2 else LEFT)
        assert not info["collapsed"]
    assert trunc and not term


def test_floor_risk_causes_collapses_at_full_stability():
    env = MineEnv("m1_stabilna", step_limit=300, oil_start=1000, collapse_floor=0.05)
    collapses = 0
    for seed in range(20):
        env.reset(seed=seed)
        done, i = False, 0
        while not done:
            _, _, term, trunc, info = env.step(RIGHT if i % 2 else LEFT)
            done, i = term or trunc, i + 1
        collapses += info["collapsed"]
    assert collapses == 20          # przy 5% na krok zawał przed 300 krokami jest pewny w praktyce


def test_no_collapse_on_exit_step():
    env, _, _ = make(collapse_prob=1.0)
    _, r, term, _, info = env.step(UP)
    assert term and info["reason"] == "exit" and not info["collapsed"]


# --------------------------------------------------------- olej, limit
def test_running_out_of_oil_terminates_with_penalty():
    env, _, _ = make(**SAFE, oil_start=3)
    for a in (RIGHT, LEFT):
        _, _, term, _, _ = env.step(a)
        assert not term
    _, r, term, trunc, info = env.step(RIGHT)
    cfg = env.config
    assert term and not trunc and info["reason"] == "out_of_oil" and env.oil == 0
    assert r == pytest.approx(cfg.reward_step + cfg.reward_out_of_oil)
    assert cfg.reward_out_of_oil == -20.0


def test_step_limit_sets_truncated():
    env, _, _ = make(**SAFE, step_limit=5)
    for i in range(5):
        _, _, term, trunc, info = env.step(RIGHT if i % 2 == 0 else LEFT)
    assert trunc and not term and info["reason"] == "step_limit" and env.steps == 5


def test_oil_bins():
    env, _, _ = make()
    assert [env.oil_bin(v) for v in (100, 75, 74, 50, 49, 25, 24, 0, -5)] == [3, 3, 2, 2, 1, 1, 0, 0, 0]


# --------------------------------------------------------------- wyjście
def test_exit_with_load_gives_load_value():
    env, _, _ = make(**SAFE)
    env.step(LEFT)
    env.step(RIGHT)
    _, r, term, trunc, info = env.step(UP)
    cfg = env.config
    assert term and not trunc and info["reason"] == "exit"
    assert r == pytest.approx(cfg.reward_step + cfg.deposit_values["c"])
    assert info["collected_value"] == cfg.deposit_values["c"] and not info["collapsed"]


def test_exit_without_load():
    env, _, _ = make(**SAFE)
    _, r, term, _, info = env.step(UP)
    assert term and info["reason"] == "exit" and r == pytest.approx(env.config.reward_step)


# --------------------------------------------------- zdarzenie korytarza
def test_corridor_turns_into_rock_after_n_deposits():
    env, _, _ = make(**SAFE, corridor_event_after=2)
    env.step(LEFT)                                      # 1. złoże (1,1)
    assert env.grid[2][2] == "X" and not env.event_fired
    env.step(DOWN)                                      # (1,2)
    _, _, _, _, info = env.step(DOWN)                   # 2. złoże g (1,3) -> zdarzenie
    assert info["event"] == "corridor_collapse" and env.event_fired
    assert env.grid[2][2] == "#"
    env.step(RIGHT)                                     # (2,3)
    _, _, _, _, info = env.step(UP)                     # X jest teraz skałą
    assert info["illegal"] and env.pos == (2, 3)


def test_event_happens_once_and_depends_only_on_mask():
    env, _, _ = make(**SAFE, corridor_event_after=1)
    _, _, _, _, info = env.step(LEFT)
    assert info["event"] == "corridor_collapse"
    _, _, _, _, info = env.step(RIGHT)
    assert info["event"] is None


# ------------------------------------------------------- obserwacje / Q
def test_state_space_size_matches_report():
    env = MineEnv("m3_gleboka")                         # K = 5
    assert env.n_deposits == 5
    assert env.n_states == 12 * 11 * 6 * 4 * 2 ** 5 * 3   # ok. 3,0e5


def test_obs_index_is_unique_and_in_range():
    env = MineEnv("m1_stabilna")
    rng = np.random.default_rng(0)
    seen = {}
    for ep in range(30):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            idx = env.obs_to_index(obs)
            key = env.obs_to_key(obs)
            assert 0 <= idx < env.n_states
            assert seen.setdefault(idx, key) == key
            vec = env.obs_to_vector(obs)
            assert vec.shape == (env.vector_size,) and vec.dtype == np.float32
            assert np.all((vec >= 0) & (vec <= 1))
            assert env.observation_space.contains(obs)
            obs, _, term, trunc, _ = env.step(int(rng.integers(5)))
            done = term or trunc


def test_too_many_deposits_rejected():
    with pytest.raises(ValueError):
        MineEnv(parse_map("#E######\n#Scsgkcs\n", "za_duzo"))


def test_invalid_action_raises():
    env, _, _ = make()
    with pytest.raises(ValueError):
        env.step(5)


def test_step_before_reset_raises():
    env = MineEnv(parse_map(TINY, "t"))
    with pytest.raises(RuntimeError):
        env.step(UP)


def test_ansi_render():
    env = MineEnv(parse_map(TINY, "t"), render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    assert "@" in text and "stabilnosc" in text


# --------------------------------------------------------------- mapy
@pytest.mark.parametrize("path", list_maps(), ids=lambda p: p.stem)
def test_map_files_follow_report(path):
    m = load_map(path)
    assert (m.width, m.height) == (12, 11)
    assert m.exit[1] == 0 and 1 <= len(m.deposits) <= 5
    reach = reachable(m.rows, m.start)
    assert m.exit in reach
    assert all((x, y) in reach for x, y, _ in m.deposits)
    MineEnv(m)                                          # konfiguracja przechodzi walidację


def test_corridor_map_has_alternative_route():
    m = load_map("m4_zawal_korytarza")
    assert m.collapse_cells
    rows = [r.replace("X", "#") for r in m.rows]
    reach = reachable(rows, m.start)
    assert all((x, y) in reach for x, y, _ in m.deposits)


def test_map_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_map("##E#\n#S.\n")          # różne długości
    with pytest.raises(ValueError):
        parse_map("####\n#SE#\n")         # wyjście poza wierszem 0
    with pytest.raises(ValueError):
        parse_map("#E#\n#T#\n#S#\n")      # nieznany znak


# ------------------------------------------------------------ konfiguracja
def test_defaults_match_report():
    cfg = GameConfig()
    assert (cfg.reward_step, cfg.reward_illegal, cfg.reward_collapse, cfg.reward_out_of_oil) \
        == (-0.1, -0.5, -20.0, -20.0)
    assert (cfg.oil_start, cfg.oil_bins, cfg.props_start, cfg.props_max) == (100, 4, 2, 2)
    assert (cfg.stability_max, cfg.corridor_event_after, cfg.step_limit) == (5, 3, 300)
    assert cfg.deposit_values == {"c": 3, "s": 9, "g": 18, "k": 30}


def test_map_header_overrides_config():
    env = MineEnv("m2_krucha")
    assert env.config.stability_start == 3 and env.config.props_start == 1
    env = MineEnv("m2_krucha", stability_start=4)   # jawne nadpisanie wygrywa
    assert env.config.stability_start == 4


def test_config_from_json(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"collapse_base": 0.05, "reward_collapse": -50}), encoding="utf-8")
    cfg = GameConfig.from_file(path)
    assert cfg.collapse_base == 0.05 and cfg.reward_collapse == -50
    assert GameConfig.from_dict(cfg.to_dict()) == cfg


def test_config_rejects_unknown_key():
    with pytest.raises(KeyError):
        GameConfig().replace(nie_ma_takiego=1)


# ------------------------------------------------------- losowe mapy
def test_random_maps_change_between_episodes_and_follow_seed():
    env = MineEnv(random_maps=True)
    env.reset(seed=1)
    rows_a = env.map.rows
    env.reset(seed=2)
    rows_b = env.map.rows
    env.reset(seed=1)
    assert env.map.rows == rows_a != rows_b
    assert (env.width, env.height, env.n_deposits) == (12, 11, 5)
    assert env.observation_space.contains(env._obs())


def test_random_maps_trajectory_is_reproducible():
    def trace(seed):
        env = MineEnv(random_maps=True)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(0)
        out = [env.map.rows]
        for _ in range(200):
            obs, r, term, trunc, info = env.step(int(rng.integers(5)))
            out.append((tuple(obs), r, info["reason"]))
            if term or trunc:
                break
        return out
    assert trace(5) == trace(5)


def test_local_view_observation():
    env, obs, _ = make(**SAFE, obs_mode="local", view_radius=1)
    # TINY: wokół S (2,1): wiersz 0 = '#','E','#'; wiersz 1 = 'c','S','.'; wiersz 2 = '.','X','.'
    assert obs[7:].tolist() == [0, 3, 0, 2, 1, 1, 1, 4, 1]
    assert env.observation_space.contains(obs)
    assert env.obs_to_vector(obs).shape == (env.vector_size,)
    obs, *_ = env.step(LEFT)             # wydobycie c: pole staje się korytarzem
    assert obs[7:].tolist() == [0, 0, 3, 0, 1, 1, 0, 1, 4]


def test_map_seed_option_separates_map_from_game():
    env = MineEnv(random_maps=True)
    env.reset(seed=1, options={"map_seed": 42})
    rows = env.map.rows
    env.reset(seed=2, options={"map_seed": 42})
    assert env.map.rows == rows                     # ta sama kopalnia, inne losowanie zawałów
    env.reset(seed=1, options={"map_seed": 43})
    assert env.map.rows != rows
