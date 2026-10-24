"""Nagrywanie deterministycznych epizodów do zapętlonych GIF-ów."""
import os
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miner_rl.render.cli import parser, create, positive
from miner_rl.agents.loader import make_agent
from miner_rl.agents.tabular import act_in_env
import imageio.v2 as imageio


def _safe(value: str) -> str:
    return re.sub(r'[^\w.-]+', '_', value).strip('._') or 'agent'


def main(argv=None) -> list[Path]:
    p = parser('Nagrywanie epizodów do GIF', agent=True, scale=1)
    p.add_argument('--out', default='results/gifs/', help='Katalog wyjściowy')
    p.add_argument('--only-failed', action='store_true', help='Zapisuj tylko niepowodzenia')
    p.add_argument('--max-tries', type=positive, default=200, help='Limit prób szukania niepowodzeń')
    args = p.parse_args(argv)
    if args.seed is None:
        args.seed = 0
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for attempt in range(args.max_tries if args.only_failed else args.episodes):
        env, renderer = create(args)
        seed = args.seed+attempt
        try:
            obs, _ = env.reset(seed=seed)
            if args.random:
                renderer.seed = seed
            agent = make_agent(args.agent, env, seed, args.threshold)
            getattr(agent, 'reset', lambda: None)()
            frames = [renderer.frame_array(renderer.draw())]
            while True:
                obs, _, terminated, truncated, info = env.step(act_in_env(agent, env, obs))
                frames.append(renderer.frame_array(renderer.draw()))
                if terminated or truncated:
                    break
            if args.only_failed and info['reason'] == 'exit':
                continue
            frames.extend(renderer.frame_array(frame) for frame in renderer.end_frames())
            path = out / f'{_safe(env.map.name)}_{_safe(agent.name)}_seed{seed}_{info["reason"]}.gif'
            # Wtyczka Pillow przyjmuje czas w milisekundach, nie w sekundach.
            imageio.mimsave(path, frames, format='GIF', duration=1000/args.fps, loop=0)
            paths.append(path)
            print(path)
            if len(paths) == args.episodes:
                break
        finally:
            renderer.close()
            env.close()
    if len(paths) < args.episodes:
        print(f'Zapisano {len(paths)} z {args.episodes} epizodów: osiągnięto limit prób.')
    return paths


if __name__ == '__main__':
    main()
