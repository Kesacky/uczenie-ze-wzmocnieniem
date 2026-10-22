"""Gra człowieka: strzałki — ruch, P — stempel, R — ponownie."""
from __future__ import annotations
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miner_rl.render.cli import parser, create, resolve_scale
from miner_rl.render.ui import make_buttons, hit_test, draw_menu, draw_end_screen, summary


def choose(surface, choices, title):
    """Zwraca wybór myszy lub klawiatury; Esc wraca o jeden poziom."""
    import pygame
    window = pygame.display.set_mode(surface.get_size())
    buttons = make_buttons(surface, choices, .33)
    selected = 0
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.MOUSEMOTION:
                selected = next((i for i, b in enumerate(buttons) if b.rect.collidepoint(event.pos)), -1)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                key = hit_test(buttons, event.pos)
                if key is not None:
                    return key
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 'back'
                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    selected = (selected + (1 if event.key == pygame.K_DOWN else -1)) % len(buttons)
                if pygame.K_1 <= event.key <= pygame.K_9 and event.key-pygame.K_1 < len(buttons):
                    selected = event.key-pygame.K_1
                if event.key == pygame.K_RETURN and selected >= 0:
                    return buttons[selected].key
        draw_menu(surface, buttons, title, pygame.mouse.get_pos(), selected)
        window.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)


def menu(args):
    import pygame
    from miner_rl.env.maps import list_maps, load_map
    scale = resolve_scale(args.scale, 16, 16)
    surface = pygame.Surface((round(512*scale), round(600*scale)))
    while True:
        choice = choose(surface, [('fixed', 'Stała mapa'), ('random', 'Losowa mapa'), ('quit', 'Wyjście')],
                        'Górnik w niestabilnej kopalni')
        if choice in ('quit', 'back'):
            return False
        if choice == 'random':
            args.random, args.map = True, None
            return True
        maps = list_maps()
        choices = [(str(i), f'{i+1}. {load_map(path).name} — {load_map(path).description}') for i, path in enumerate(maps)]
        choice = choose(surface, choices+[('back', 'Wróć')], 'Wybierz stałą mapę')
        if choice == 'quit':
            return False
        if choice != 'back':
            args.map, args.random = str(maps[int(choice)]), False
            return True


def main(argv=None):
    p = parser('Gra człowieka w kopalni')
    p.add_argument('--script', help='Lista akcji oddzielonych przecinkami; bez okna')
    p.add_argument('--no-menu', action='store_true', help='Pomiń menu startowe')
    p.add_argument('--save-frame', help='Zapisz ostatnią klatkę do PNG')
    args = p.parse_args(argv)
    if args.script is not None:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    import pygame
    pygame.display.init()
    pygame.display.set_caption('Górnik w niestabilnej kopalni')
    records = {}
    import secrets
    fixed_seed = args.seed is not None or args.script is not None
    base_seed = args.seed if args.seed is not None else 0

    def fresh():
        """Nowe ziarno: losowe, chyba że użytkownik podał --seed (wtedy kolejne liczby)."""
        nonlocal base_seed
        if fixed_seed:
            base_seed += 1
            return base_seed - 1
        return secrets.randbelow(2**31)

    map_seed = fresh()        # tryb losowy: która kopalnia
    game_seed = fresh()       # przebieg gry (losowanie zawałów)
    if args.script is None and not args.map and not args.random and not args.no_menu:
        if not menu(args):
            pygame.display.quit()
            return 0
    try:
        while True:
            env, renderer = create(args)
            env.reset(seed=game_seed, options={"map_seed": map_seed})
            if args.random:
                renderer.seed = map_seed
            try:
                if args.script is not None:
                    try:
                        actions = [int(a.strip()) for a in args.script.split(',') if a.strip()]
                    except ValueError:
                        p.error('Skrypt musi zawierać liczby od 0 do 4.')
                    if any(a not in range(5) for a in actions):
                        p.error('Akcje muszą być w zakresie od 0 do 4.')
                    for action in actions:
                        _, _, terminated, truncated, _ = env.step(action)
                        renderer.draw()
                        if terminated or truncated:
                            break
                    renderer.surface.blit(renderer.end_frames()[-1], (0, 0))
                    if args.save_frame:
                        path = Path(args.save_frame)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        pygame.image.save(renderer.surface, str(path))
                    return 0
                renderer.open_window()
                action = play_episode(env, renderer, records, args.random)
            finally:
                renderer.close()
                env.close()
            if action == 'quit':
                return 0
            game_seed = fresh()                     # każda gra (także „Zagraj ponownie”) losuje zawały od nowa
            if action == 'new':
                map_seed = fresh()                  # nowa kopalnia
            if action == 'menu':
                if not menu(args):
                    return 0
                map_seed = fresh()
    finally:
        pygame.display.quit()


def play_episode(env, renderer, records, random):
    import pygame
    clock = pygame.time.Clock()
    keys = {pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 3, pygame.K_p: 4}
    shortcuts = {pygame.K_r: 'retry', pygame.K_n: 'new', pygame.K_m: 'menu', pygame.K_ESCAPE: 'quit'}
    done, base, data = False, None, None
    buttons = []
    selected = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if done and event.type == pygame.MOUSEMOTION:
                selected = next((i for i, button in enumerate(buttons) if button.rect.collidepoint(event.pos)), -1)
            if done and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                choice = hit_test(buttons, event.pos)
                if choice:
                    return choice
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                return 'quit'
            if done:
                if event.key in shortcuts and (event.key != pygame.K_n or random):
                    return shortcuts[event.key]
                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    selected = (selected+(1 if event.key == pygame.K_DOWN else -1)) % len(buttons)
                if event.key == pygame.K_RETURN and selected >= 0:
                    return buttons[selected].key
            elif event.key == pygame.K_r:
                return 'retry'
            elif event.key in keys:
                _, _, terminated, truncated, _ = env.step(keys[event.key])
                done = terminated or truncated
                if done:
                    for frame in renderer.collapse_frames() if env.last_info.get('reason') == 'collapse' else []:
                        renderer.present(frame)
                        pygame.event.pump()
                        clock.tick(12)
                    key = 'losowa' if random else env.map.name
                    new = key not in records or env.total_reward > records[key]
                    records[key] = max(env.total_reward, records.get(key, float('-inf')))
                    data = summary(env, records[key], new)
                    base = renderer.draw().copy()
                    choices = [('retry', 'Zagraj ponownie (R)')]
                    if random:
                        choices.append(('new', 'Nowa mapa (N)'))
                    choices += [('menu', 'Menu (M)'), ('quit', 'Wyjście (Esc)')]
                    buttons = make_buttons(base, choices)
        frame = base.copy() if done else renderer.draw()
        if done:
            draw_end_screen(frame, data, buttons, mouse_pos=pygame.mouse.get_pos(), selected=selected)
        renderer.present(frame)
        clock.tick(30)


if __name__ == '__main__':
    raise SystemExit(main())
