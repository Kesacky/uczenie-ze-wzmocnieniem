"""Podgląd agenta z pauzą i regulacją prędkości."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miner_rl.render.cli import parser, create
from miner_rl.agents.loader import make_agent


def main(argv=None) -> int:
    p = parser('Podgląd agenta: spacja pauzuje, +/- zmienia tempo', agent=True)
    p.add_argument('--headless', action='store_true', help='Rysuj bez okna')
    args = p.parse_args(argv)
    if args.headless:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    import pygame
    env, renderer = create(args)
    fps, paused = args.fps, False
    clock = pygame.time.Clock()

    def controls():
        nonlocal fps, paused
        if args.headless:
            return True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fps += 1
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps = max(1, fps-1)
        return True

    try:
        if not args.headless:
            renderer.open_window()
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed+episode)
            agent = make_agent(args.agent, env, args.seed+episode, args.threshold)
            getattr(agent, 'reset', lambda: None)()
            done = False
            renderer.draw()
            while not done:
                if not controls():
                    return 0
                if not paused:
                    obs, _, terminated, truncated, _ = env.step(agent.act(obs, greedy=True))
                    done = terminated or truncated
                frame = renderer.draw()
                if not args.headless:
                    renderer.present(frame)
                    clock.tick(fps if not paused else 30)
            for frame in renderer.end_frames():
                while True:
                    if not controls():
                        return 0
                    if not args.headless:
                        renderer.present(frame)
                        clock.tick(fps if not paused else 30)
                    if not paused:
                        break
            print(f'Zwrot: {env.total_reward:.2f}, długość: {env.steps}, przyczyna: {env.last_info["reason"]}')
        return 0
    finally:
        renderer.close()
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
