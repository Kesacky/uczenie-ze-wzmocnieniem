"""Interfejs i skala: testy na powierzchniach bez prawdziwego okna."""
import os
from pathlib import Path
import subprocess
import sys
import pygame
import pytest
from miner_rl.env import MineEnv
from miner_rl.render import AssetPack, Renderer
from miner_rl.render.cli import fit_scale, parser, create
from miner_rl.render.ui import Button, hit_test, make_buttons, draw_menu, draw_end_screen, summary


@pytest.mark.parametrize('screen,expected', [((1920, 1080), 1.5), ((1280, 720), 1.0),
                                              ((800, 600), .75), ((200, 160), .5), ((4000, 3000), 2)])
def test_fit(screen, expected):
    assert fit_scale(*screen, 16, 16) == expected


def test_fractional():
    pack = AssetPack(scale=1.5)
    assert pack.get('tiles/coal').get_size() == (48, 48)
    env = MineEnv(random_maps=True)
    env.reset(seed=4)
    r = Renderer(env, scale=1.5)
    assert r.draw().get_size() == (env.width*48, env.height*48+132)
    assert AssetPack(scale=.1).get('tiles/coal').get_size() == (8, 8)


def test_hit():
    buttons = [Button(pygame.Rect(10, 10, 30, 20), 'Test', 'a')]
    assert hit_test(buttons, (10, 10)) == 'a'
    assert hit_test(buttons, (40, 10)) is None


@pytest.mark.parametrize('reason', ['exit', 'collapse', 'out_of_oil', 'step_limit'])
def test_screens(reason):
    env = MineEnv()
    env.reset(seed=1)
    r = Renderer(env, scale=.5)
    env.last_info['reason'] = reason
    frame = r.draw().copy()
    buttons = make_buttons(frame, [('retry', 'Zagraj ponownie'), ('quit', 'Wyjście')])
    draw_menu(frame, buttons)
    draw_end_screen(frame, summary(env, 10, True), buttons, mouse_pos=buttons[0].rect.center)
    assert len(r.end_frames(3)) == 3


def test_random_scripts(tmp_path):
    from scripts.play import main
    path = tmp_path/'klatka.png'
    assert main(['--script', '0,2,3', '--random', '--scale', '1.5', '--save-frame', str(path)]) == 0
    assert pygame.image.load(str(path)).get_width() == 12 * 48     # mapa 12 pól × kafelek 32·1,5
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, 'scripts/watch.py', '--headless', '--random',
                             '--episodes', '1', '--no-textures', '--obs-mode', 'local'],
                            cwd=root, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    from scripts.record import main as record
    paths = record(['--random', '--episodes', '1', '--scale', '.5', '--out', str(tmp_path), '--seed', '3'])
    assert len(paths) == 1
    assert paths[0].name.startswith('losowa_random_seed3_')


def test_auto_dummy():
    args = parser('Test').parse_args(['--random', '--obs-mode', 'local', '--view-radius', '3'])
    env, renderer = create(args)
    assert renderer.scale == 1
    env.reset(seed=5)
    assert env.map.name == 'losowa'


def test_launcher():
    content = Path('graj.bat').read_bytes()
    content.decode('ascii')
    assert b'\r\n' in content and b'\n' not in content.replace(b'\r\n', b'')


def test_menu_keyboard_and_mouse(monkeypatch):
    from scripts.play import choose
    surface = pygame.Surface((512, 600))
    choices = [('fixed', 'Stała mapa'), ('random', 'Losowa mapa'), ('quit', 'Wyjście')]
    pygame.display.init()
    monkeypatch.setattr(pygame.event, 'get', lambda: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2),
                                                    pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)])
    assert choose(surface, choices, 'Menu') == 'random'
    buttons = make_buttons(surface, choices, .33)
    monkeypatch.setattr(pygame.event, 'get', lambda: [pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                              button=1, pos=buttons[0].rect.center)])
    assert choose(surface, choices, 'Menu') == 'fixed'
    pygame.display.quit()


def test_end_screen_session_record(monkeypatch):
    from scripts.play import play_episode
    env = MineEnv(collapse_prob=1, stability_start=1)
    env.reset(seed=3)
    renderer = Renderer(env, scale=.5)
    renderer.open_window()
    events = iter([[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)],
                   [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)]])
    monkeypatch.setattr(pygame.event, 'get', lambda: next(events))
    monkeypatch.setattr(renderer, 'collapse_frames', lambda: [])
    records = {}
    try:
        assert play_episode(env, renderer, records, False) == 'retry'
        assert records[env.map.name] == env.total_reward
    finally:
        renderer.close()
