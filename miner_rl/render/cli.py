"""Wspólne argumenty i inicjalizacja programów graficznych."""
import argparse
import math
import pygame
from miner_rl.env import MineEnv
from miner_rl.render import Renderer


def positive(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError('Wartość musi być dodatnia.')
    return value


def scale_value(raw):
    if raw == 'auto':
        return raw
    try:
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError
        return value
    except ValueError:
        raise argparse.ArgumentTypeError('Skala musi być dodatnią liczbą albo auto.')


def fit_scale(screen_w, screen_h, map_w, map_h, tile=32, hud_base=88,
              margin=0.9, max_scale=2.0, min_scale=0.5):
    """Największa pasująca ćwiartka skali; minimum, gdy ekran jest za mały."""
    for quarter in range(math.floor(max_scale*4), math.ceil(min_scale*4)-1, -1):
        scale = quarter/4
        size = max(8, round(tile*scale))
        if map_w*size <= screen_w*margin and map_h*size+round(hud_base*scale) <= screen_h*margin:
            return scale
    return min_scale


def resolve_scale(value, width, height):
    if value != 'auto':
        return value
    try:
        pygame.display.init()
        info = pygame.display.Info()
        if pygame.display.get_driver() != 'dummy' and info.current_w > 0 and info.current_h > 0:
            return fit_scale(info.current_w, info.current_h, width, height)
    except pygame.error:
        pass
    return 1.0


def parser(description, agent=False, scale='auto'):
    p = argparse.ArgumentParser(description=description)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--map', help='Nazwa lub ścieżka mapy')
    mode.add_argument('--random', action='store_true', help='Losuj mapę każdego epizodu')
    p.add_argument('--obs-mode', choices=['state', 'local', 'hint'], default='state')
    p.add_argument('--hint-kind', choices=['compass', 'path'], default='compass',
                   help='Podpowiedź narratora przy --obs-mode hint (ho1 = compass, ho2 i ho3 = path)')
    p.add_argument('--hint-memory', type=int, nargs='?', const=1, default=0, choices=[0, 1, 2],
                   help='Pamięć sąsiednich pól w obserwacji narratora: 1 = odwiedzone (modele ho3), '
                        '2 = liczba wejść 0/1/2+ (modele ho4)')
    p.add_argument('--view-radius', type=positive, default=2)
    p.add_argument('--oil-bins', type=int, default=None,
                   help='Przedziały oleju w obserwacji (jak przy treningu agenta; domyślnie z konfiguracji)')
    p.add_argument('--seed', type=int, default=None,
                   help='Ziarno losowości (play: brak = losowe przy każdej grze; watch/record: brak = 0)')
    p.add_argument('--scale', type=scale_value, default=scale, help='Skala grafiki lub auto')
    p.add_argument('--no-textures', action='store_true', help='Użyj zastępników tekstur')
    p.add_argument('--no-lamp', action='store_true', help='Wyłącz ciemność')
    if agent:
        p.add_argument('--agent', default='random', help='Nazwa, plik pickle lub moduł:funkcja')
        p.add_argument('--threshold', type=float, default=6, help='Próg powrotu')
        p.add_argument('--fps', type=positive, default=8, help='Klatki na sekundę')
        p.add_argument('--episodes', type=positive, default=3, help='Liczba epizodów')
    return p


def create(args):
    overrides = {} if getattr(args, 'oil_bins', None) is None else {'oil_bins': args.oil_bins}
    env = MineEnv(args.map or 'm1_stabilna', random_maps=args.random,
                  obs_mode=args.obs_mode, view_radius=args.view_radius,
                  hint_kind=getattr(args, 'hint_kind', 'compass'),
                  hint_memory=getattr(args, 'hint_memory', 0), **overrides)
    renderer = Renderer(env, scale=resolve_scale(args.scale, env.width, env.height),
                        use_textures=not args.no_textures, lamp=not args.no_lamp)
    if args.random:
        renderer.seed = args.seed
    return env, renderer
