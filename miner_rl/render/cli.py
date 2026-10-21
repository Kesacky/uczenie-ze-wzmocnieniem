"""Wspólne argumenty i inicjalizacja programów graficznych."""
import argparse
from miner_rl.env import MineEnv
from miner_rl.render import Renderer


def positive(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError('Wartość musi być dodatnia.')
    return value


def parser(description: str, agent: bool = False, scale: int = 2):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--map', default='m1_stabilna', help='Nazwa lub ścieżka mapy')
    p.add_argument('--seed', type=int, default=0, help='Ziarno losowości')
    p.add_argument('--scale', type=positive, default=scale, help='Skala grafiki')
    p.add_argument('--no-textures', action='store_true', help='Użyj zastępników tekstur')
    p.add_argument('--no-lamp', action='store_true', help='Wyłącz ciemność')
    if agent:
        p.add_argument('--agent', default='random', help='Nazwa, plik pickle lub moduł:funkcja')
        p.add_argument('--threshold', type=float, default=6, help='Próg powrotu')
        p.add_argument('--fps', type=positive, default=8, help='Klatki na sekundę')
        p.add_argument('--episodes', type=positive, default=3, help='Liczba epizodów')
    return p


def create(args):
    env = MineEnv(args.map)
    renderer = Renderer(env, scale=args.scale, use_textures=not args.no_textures, lamp=not args.no_lamp)
    return env, renderer
