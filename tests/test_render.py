"""Szybkie testy grafiki bez okna oraz programów użytkowych."""
import pickle
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import pytest

from miner_rl.agents.base import Agent
from miner_rl.agents.baselines import RandomAgent, ThresholdAgent
from miner_rl.agents.loader import make_agent
from miner_rl.env import MineEnv, list_maps
from miner_rl.render import AssetPack, Renderer
from miner_rl.render.assets import ASSET_NAMES


@pytest.fixture
def textures(tmp_path):
    for name in ('tiles/coal', 'tiles/rock_0', 'sprites/miner_right_0', 'ui/hud_frame'):
        path = tmp_path / (name+'.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGBA', (32, 32), (123, 85, 42, 255)).save(path)
    return tmp_path


@pytest.mark.parametrize('enabled', [False, True])
def test_fallbacks(tmp_path, enabled):
    pack = AssetPack(tmp_path, use_textures=enabled)
    for name in ASSET_NAMES:
        assert pack.get(name).get_size() == (64, 64)
        assert pack.get(name) is pack.get(name)
    assert pack.missing_names == set(ASSET_NAMES)
    assert not pack.textured_names


def test_textures(textures):
    pack = AssetPack(textures)
    for name in ('tiles/coal', 'tiles/rock_0', 'sprites/miner_right_0', 'ui/hud_frame'):
        surface = pack.get(name)
        assert surface.get_size() == (64, 64)
        assert surface.get_at((0, 0)) == (123, 85, 42, 255)
        assert name in pack.textured_names
    (textures/'tiles/water.png').write_bytes(b'niepoprawny PNG')
    assert pack.get('tiles/water').get_size() == (64, 64)
    assert 'tiles/water' in pack.missing_names


def snapshot(env):
    """Stan oraz generator środowiska, bez współdzielonych referencji."""
    return pickle.dumps((env.grid, env.pos, env.facing, env.stability, env.oil,
                         env.load_value, env.props, env.prop_cells, env.steps,
                         env.last_action, env.last_dug, env.last_info, env.total_reward,
                         env.np_random.bit_generator.state))


@pytest.mark.parametrize('map_name', list_maps())
@pytest.mark.parametrize('textured', [False, True])
def test_draw(map_name, textured, textures):
    env = MineEnv(map_name)
    obs, _ = env.reset(seed=2)
    key = env.obs_to_key(obs)
    before = snapshot(env)
    renderers = [Renderer(env, assets_dir=textures, use_textures=textured) for _ in range(2)]
    for _ in range(3):
        frames = [r.frame_array(r.draw()) for r in renderers]
        assert np.array_equal(*frames)
        assert frames[0].shape == (env.height*64+176, env.width*64, 3)
        assert frames[0].dtype == np.uint8
    assert snapshot(env) == before
    assert env.obs_to_key(obs) == key


def test_effects():
    env = MineEnv(stability_start=1, collapse_prob=1.0)
    env.reset(seed=3)
    a, b = Renderer(env, use_textures=False), Renderer(env, use_textures=False)
    for _ in range(3):
        assert np.array_equal(a.frame_array(a.draw()), b.frame_array(b.draw()))
    _, _, terminated, _, info = env.step(3)      # ruch w prawo po korytarzu
    assert terminated and info['reason'] == 'collapse'
    before = snapshot(env)
    frames = a.collapse_frames()
    assert len(frames) == 8
    assert not np.array_equal(a.frame_array(frames[0]), a.frame_array(frames[-1]))
    assert len(a.end_frames(3)) == 3
    assert snapshot(env) == before


def test_dig_animation():
    env = MineEnv(collapse_prob=0)
    env.reset(seed=0)
    env.last_action, env.last_dug = 2, (1, 1)   # właśnie wydobyte złoże
    renderer = Renderer(env, lamp=False, hud=False, use_textures=False)
    before = snapshot(env)
    renderer.draw()
    assert renderer._dig == 1
    renderer.draw()
    assert renderer._dig == 0
    renderer.draw()
    assert renderer._dig == 0
    assert snapshot(env) == before


def test_agent_loader(tmp_path, monkeypatch):
    env = MineEnv()
    assert isinstance(make_agent('random', env, seed=1), RandomAgent)
    assert isinstance(make_agent('threshold', env), ThresholdAgent)
    module = tmp_path/'pomocniczy_agent.py'
    module.write_text('from miner_rl.agents.baselines import RandomAgent\ndef create(env):\n    return RandomAgent(seed=4)\n', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    assert isinstance(make_agent('pomocniczy_agent:create', env), Agent)
    with pytest.raises(ValueError, match='Nieznana'):
        make_agent('nieznany', env)
    with pytest.raises(ValueError):
        make_agent('pomocniczy_agent:brak', env)
    path = tmp_path/'agent.pkl'
    path.write_bytes(pickle.dumps(RandomAgent(seed=0)))
    assert isinstance(make_agent(str(path), env), RandomAgent)
    path.write_bytes(pickle.dumps(123))
    with pytest.raises(ValueError):
        make_agent(str(path), env)


@pytest.mark.parametrize('script,args', [
    ('play.py', ['--script', '2,3,4,0']),
    ('watch.py', ['--headless', '--episodes', '1', '--agent', 'random']),
    ('watch.py', ['--headless', '--episodes', '1', '--agent', 'threshold']),
])
def test_programs(script, args):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root/'scripts'/script), '--scale', '1',
                             '--no-textures', *args], cwd=root, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr


def test_record(tmp_path):
    from scripts.record import main
    paths = main(['--out', str(tmp_path), '--episodes', '1', '--agent', 'threshold',
                  '--no-textures', '--fps', '8'])
    assert len(paths) == 1
    frames = imageio.mimread(paths[0])
    assert len(frames) > 1
    with Image.open(paths[0]) as gif:
        assert gif.info['loop'] == 0
        assert abs(gif.info['duration']-125) <= 10
    paths2 = main(['--out', str(tmp_path/'failed'), '--episodes', '1', '--only-failed',
                   '--max-tries', '50', '--agent', 'threshold', '--threshold', '15',
                   '--no-textures', '--no-lamp'])
    assert len(paths2) == 1
    assert not paths2[0].name.endswith('_exit.gif')


def test_rubble_after_corridor_event(tmp_path):
    env = MineEnv('m4_zawal_korytarza', collapse_prob=0.0)
    env.reset(seed=0)
    renderer = Renderer(env, use_textures=False, lamp=False)
    before = renderer.frame_array(renderer.draw()).copy()
    env.event_fired = True                       # tylko do testu widoku
    for x, y in env.map.collapse_cells:
        env.grid[y][x] = '#'
    after = renderer.frame_array(renderer.draw())
    x, y = env.map.collapse_cells[0]
    t = renderer.tile
    assert not np.array_equal(before[y*t:(y+1)*t, x*t:(x+1)*t], after[y*t:(y+1)*t, x*t:(x+1)*t])


def test_loader_reads_tabular_model(tmp_path):
    from miner_rl.agents.tabular import QLearningAgent, EpsilonSchedule, train
    env = MineEnv('m1_stabilna')
    agent = QLearningAgent(n_actions=env.action_space.n, epsilon=EpsilonSchedule(1.0, 0.1, 20), seed=0)
    train(env, agent, episodes=30, seed=0)
    path = tmp_path / 'q.pkl'
    agent.save(path)
    loaded = make_agent(str(path), env)
    obs, _ = env.reset(seed=0)
    assert loaded.act(obs, greedy=True) in range(env.action_space.n)
    from scripts.record import main
    paths = main(['--out', str(tmp_path / 'gif'), '--episodes', '1', '--agent', str(path),
                  '--no-textures', '--no-lamp'])
    assert len(paths) == 1 and paths[0].exists()
