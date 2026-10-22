"""Deterministyczny widok kopalni; wszystkie odwołania do środowiska są odczytem."""
from __future__ import annotations
import numpy as np
import pygame
from .assets import AssetPack
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miner_rl.env import MineEnv


class Renderer:
    """Rysuje na własnej powierzchni, opcjonalnie prezentowanej w oknie."""

    def __init__(self, env: MineEnv, scale: float = 2, assets: AssetPack | None = None,
                 assets_dir=None, use_textures: bool = True, lamp: bool = True,
                 hud: bool = True) -> None:
        pygame.font.init()
        self.env = env
        self.assets = assets or AssetPack(assets_dir, scale=scale, use_textures=use_textures)
        self.scale = self.assets.scale
        self.tile = max(8, round(self.assets.tile * self.scale))
        self.scale = self.tile / self.assets.tile
        self.hud_height = round(88 * self.scale) if hud else 0
        self.surface = pygame.Surface((env.width*self.tile, env.height*self.tile+self.hud_height))
        self.font = pygame.font.Font(pygame.font.match_font('dejavusans'), max(8, round(13*self.scale)))
        self.lamp, self.hud = lamp, hud
        self.frame = 0
        self._previous_pos = env.pos
        self._walk = 0
        self._step = None
        self._dig = 0
        self._notice = 0
        self._window = None

    def _text(self, s: pygame.Surface, text: str, pos: tuple[int, int]) -> None:
        s.blit(self.font.render(text, True, (245, 232, 204)), pos)

    def draw(self) -> pygame.Surface:
        e, t = self.env, self.tile
        if self._step != e.steps:
            self._dig = 2 if e.last_dug is not None else 0   # wejście na złoże = wydobycie
            self._notice = 8 if e.last_info.get('event') == 'corridor_collapse' else 0
            self._step = e.steps
        if e.pos != self._previous_pos:
            self._walk = 1-self._walk
            self._previous_pos = e.pos
        world = pygame.Surface((e.width*t, e.height*t))
        names = {'c': 'coal', 's': 'silver', 'g': 'gold', 'k': 'crystal',
                 'X': 'corridor_cracked', 'E': 'exit'}
        # Po zdarzeniu pola X są w logice zwykłą skałą; rysujemy je jako gruz.
        rubble = set(e.map.collapse_cells) if e.event_fired else set()
        for y, row in enumerate(e.grid):
            for x, ch in enumerate(row):
                p = (x*t, y*t)
                world.blit(self.assets.get('tiles/corridor'), p)
                if (x, y) in rubble:
                    name = 'rubble'
                elif ch == '#':
                    name = f'rock_{(x*17+y*31)%3}'
                    if y == 0:
                        self.assets.get('tiles/surface')
                        if 'tiles/surface' in self.assets.textured_names:
                            name = 'surface'
                else:
                    name = names.get(ch)
                if name:
                    world.blit(self.assets.get('tiles/'+name), p)
                if (x, y) in e.prop_cells:
                    world.blit(self.assets.get('tiles/prop'), p)
        direction = ('up', 'down', 'left', 'right')[e.facing]
        name = f'sprites/miner_{direction}_{self._walk}'
        if self._dig:
            name = f'sprites/miner_dig_{2-self._dig}'
        elif e.load_value > 0:
            self.assets.get('sprites/miner_loaded')
            if 'sprites/miner_loaded' in self.assets.textured_names:
                name = 'sprites/miner_loaded'
        sprite = self.assets.get(name)
        if e.facing == 2 and ('dig' in name or 'loaded' in name):
            sprite = pygame.transform.flip(sprite, True, False)
        x, y = e.pos
        world.blit(sprite, (x*t, y*t))
        if e.load_value > 0 and name != 'sprites/miner_loaded':
            sack = pygame.transform.scale(self.assets.get('ui/sack'), (t//3, t//3))
            world.blit(sack, (x*t, y*t+t//2))
        if self._dig and e.last_dug is not None:
            dx, dy = e.last_dug
            world.blit(self.assets.get('sprites/flash'), (dx*t, dy*t))
        if e.stability <= 1:
            rng = np.random.default_rng(0)
            cells = [(cx, cy) for cy, row in enumerate(e.grid) for cx, ch in enumerate(row)
                     if ch in '.SEX' and abs(cx-x)+abs(cy-y) <= 3]
            if cells:
                for index in rng.integers(len(cells), size=4):
                    cx, cy = cells[index]
                    world.blit(self.assets.get(f'sprites/dust_{self.frame%3}'),
                               (cx*t, cy*t+(self.frame*3*self.scale)%t-t//2))
        if self.lamp:
            yy, xx = np.mgrid[:e.height*t, :e.width*t]
            radius = (3.0+3.0*max(0, min(1, e.oil/max(1, e.config.oil_start))))*t
            distance = np.sqrt((xx-(x+.5)*t)**2+(yy-(y+.5)*t)**2)
            darkness = pygame.Surface(world.get_size(), pygame.SRCALPHA)
            darkness.fill((20, 12, 5, 0))
            alpha = pygame.surfarray.pixels_alpha(darkness)
            # Do promienia jasno, dalej łagodne przejście w ciemność (maks. alfa 215).
            alpha[:] = np.clip((distance/radius)**1.5*215, 0, 215).T.astype(np.uint8)
            del alpha
            world.blit(darkness, (0, 0))
        self.surface.fill((25, 19, 16))
        shift = 2*self.scale*(1 if self.frame%2 else -1) if e.stability <= 1 else 0
        self.surface.blit(world, (shift, 0))
        if self.hud:
            self._draw_hud()
        if self._notice:
            self._text(self.surface, 'Zawalił się korytarz!', (8*self.scale, 8*self.scale))
        self._dig = max(0, self._dig-1)
        self._notice = max(0, self._notice-1)
        self.frame += 1
        return self.surface

    def _draw_hud(self) -> None:
        e, s, t = self.env, self.scale, self.tile
        top = e.height*t
        frame = self.assets.get('ui/hud_frame')
        # Dziewięć części ramki; brzegi zachowują szerokość ośmiu pikseli bazowych.
        margin = min(round(8*s), t//3)
        width, height = self.surface.get_width(), self.hud_height
        src = (0, margin, t-margin, t)
        xs, ys = (0, margin, width-margin, width), (top, top+margin, top+height-margin, top+height)
        for iy in range(3):
            for ix in range(3):
                part = frame.subsurface((src[ix], src[iy], src[ix+1]-src[ix], src[iy+1]-src[iy]))
                self.surface.blit(pygame.transform.scale(part, (xs[ix+1]-xs[ix], ys[iy+1]-ys[iy])), (xs[ix], ys[iy]))
        color = (74, 165, 81) if e.stability >= 4 else (220, 178, 46) if e.stability >= 2 else (205, 55, 42)
        for i, (icon, value, maximum, barcolor, label) in enumerate([
            ('stability', e.stability, e.config.stability_max, color, 'Stabilność'),
            ('oil', e.oil, e.config.oil_start, (218, 153, 51), 'Olej')]):
            left = 10*s + i*(width//2)          # dwie kolumny niezależnie od szerokości mapy
            bar = width//2 - 44*s
            self.surface.blit(pygame.transform.scale(self.assets.get('ui/'+icon), (round(20*s), round(20*s))), (left, top+10*s))
            pygame.draw.rect(self.surface, (20, 20, 20), (left+24*s, top+25*s, bar, 6*s))
            pygame.draw.rect(self.surface, barcolor, (left+24*s, top+25*s, int(bar*max(0, min(1, value/max(1, maximum)))), 6*s))
            self._text(self.surface, f'{label}: {value}', (left+24*s, top+9*s))
        self.surface.blit(pygame.transform.scale(self.assets.get('ui/sack'), (round(18*s), round(18*s))), (10*s, top+38*s))
        self._text(self.surface, f'Ładunek: {e.load_value}  Stemple: {e.props}/{e.config.props_max}  Krok: {e.steps}', (32*s, top+38*s))
        self._text(self.surface, f'Wynik: {e.total_reward:.1f}  {e.map.name}' + (f'  Ziarno: {self.seed}' if hasattr(self, 'seed') else ''), (10*s, top+60*s))

    def collapse_frames(self, n: int = 8) -> list[pygame.Surface]:
        frames = []
        x, y = self.env.pos
        for i in range(n):
            base = self.draw().copy()
            frame = base.copy()
            area = pygame.Rect(0, 0, self.env.width*self.tile, self.env.height*self.tile)
            frame.fill((25, 19, 16), area)
            frame.blit(base, ((1 if i%2 else -1)*2*self.scale, 0), area)
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, -1), (0, 1)):
                if 0 <= x+dx < self.env.width and 0 <= y+dy < self.env.height:
                    frame.blit(self.assets.get(f'sprites/collapse_{min(3, i*4//max(1,n))}'),
                               ((x+dx)*self.tile + (1 if i%2 else -1)*2*self.scale, (y+dy)*self.tile))
            shade = pygame.Surface(frame.get_size(), pygame.SRCALPHA)
            shade.fill((0, 0, 0, int(180*(i+1)/max(1,n))))
            frame.blit(shade, (0, 0))
            self._text(frame, 'Zawał!', (10*self.scale, 10*self.scale))
            frames.append(frame)
        return frames

    def end_frames(self, n: int = 8) -> list[pygame.Surface]:
        from .ui import draw_end_screen, summary
        if n < 1:
            return []
        frames = self.collapse_frames(max(1, n//2)) if self.env.last_info.get('reason') == 'collapse' and n > 1 else []
        frame = self.draw().copy()
        draw_end_screen(frame, summary(self.env), [])
        frames.extend(frame.copy() for _ in range(n-len(frames)))
        return frames

    def frame_array(self, surface: pygame.Surface | None = None) -> np.ndarray:
        return pygame.surfarray.array3d(self.surface if surface is None else surface).transpose(1, 0, 2).copy()

    def open_window(self, caption='Górnik w niestabilnej kopalni') -> None:
        pygame.display.init()
        self._window = pygame.display.set_mode(self.surface.get_size())
        pygame.display.set_caption(caption)

    def present(self, surface: pygame.Surface) -> None:
        if self._window is None:
            raise RuntimeError('Najpierw otwórz okno.')
        self._window.blit(surface, (0, 0))
        pygame.display.flip()

    def close(self) -> None:
        if self._window is not None:
            pygame.display.quit()
            self._window = None
