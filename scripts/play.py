"""Gra turowa: strzałki to ruch (wejście w złoże = wydobycie), P podepnij, R restart, Esc wyjście."""
from __future__ import annotations
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miner_rl.render.cli import parser, create


def main(argv=None) -> int:
    p = parser('Gra człowieka w kopalni')
    p.add_argument('--script', help='Lista akcji oddzielonych przecinkami; bez okna')
    args = p.parse_args(argv)
    if args.script is not None:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    import pygame
    env, renderer = create(args)
    env.reset(seed=args.seed)
    try:
        if args.script is not None:
            try:
                actions = [int(a.strip()) for a in args.script.split(',') if a.strip()]
            except ValueError:
                p.error('Skrypt musi zawierać liczby od 0 do 4.')
            if any(a not in range(5) for a in actions):
                p.error('Akcje muszą być w zakresie od 0 do 4.')
            renderer.draw()
            for action in actions:
                _, _, terminated, truncated, _ = env.step(action)
                renderer.draw()
                if terminated or truncated:
                    renderer.end_frames()
                    break
            return 0
        renderer.open_window()
        clock = pygame.time.Clock()
        keys = {pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 3,
                pygame.K_p: 4}
        done = False
        end = None
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return 0
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        env.reset(seed=args.seed)
                        renderer = _restart_renderer(renderer, env, args)
                        done, end = False, None
                    elif not done and event.key in keys:
                        _, _, terminated, truncated, _ = env.step(keys[event.key])
                        done = terminated or truncated
                        if done:
                            for frame in renderer.end_frames():
                                renderer.present(frame)
                                pygame.event.pump()
                                clock.tick(8)
                                end = frame
                            print(f'Zwrot: {env.total_reward:.2f}, długość: {env.steps}, przyczyna: {env.last_info["reason"]}')
            renderer.present(end if done else renderer.draw())
            clock.tick(30)
    finally:
        renderer.close()
        env.close()


def _restart_renderer(renderer, env, args):
    from miner_rl.render import Renderer
    renderer.close()
    result = Renderer(env, scale=args.scale, use_textures=not args.no_textures, lamp=not args.no_lamp)
    result.open_window()
    return result


if __name__ == '__main__':
    raise SystemExit(main())
