"""Tekstury kopalni i proceduralne zastępniki bez wymagania okna."""
from pathlib import Path
import pygame

TILE_NAMES = ('rock_0 rock_1 rock_2 corridor corridor_cracked coal silver gold '
              'crystal water prop exit rubble surface').split()
SPRITE_NAMES = ([f'miner_{d}_{i}' for d in ('up', 'down', 'left', 'right') for i in range(2)]
                + ['miner_dig_0', 'miner_dig_1', 'miner_loaded', 'flash']
                + [f'dust_{i}' for i in range(3)] + [f'collapse_{i}' for i in range(4)])
ASSET_NAMES = ([f'tiles/{n}' for n in TILE_NAMES] + [f'sprites/{n}' for n in SPRITE_NAMES]
               + [f'ui/{n}' for n in ('oil', 'stability', 'sack', 'hud_frame')])


class AssetPack:
    """Leniwie wczytuje i keszuje kafelki skalowane najbliższym sąsiadem."""

    def __init__(self, assets_dir=None, tile: int = 32, scale: int = 2,
                 use_textures: bool = True) -> None:
        if tile < 1 or scale < 1:
            raise ValueError('Rozmiar kafelka i skala muszą być dodatnie.')
        self.assets_dir = Path(assets_dir) if assets_dir else Path(__file__).resolve().parents[2] / 'assets'
        self.tile, self.scale = tile, scale
        self.use_textures = use_textures
        self.textured_names: set[str] = set()
        self.missing_names: set[str] = set()
        self._cache: dict[str, pygame.Surface] = {}

    def get(self, name: str) -> pygame.Surface:
        if name not in self._cache:
            try:
                if not self.use_textures:
                    raise FileNotFoundError
                surface = pygame.image.load(str(self.assets_dir / (name + '.png')))
                if pygame.display.get_surface() is not None:
                    surface = surface.convert_alpha()
                self.textured_names.add(name)
            except (OSError, ValueError, pygame.error):
                surface = self._fallback(name)
                self.missing_names.add(name)
            size = self.tile * self.scale
            self._cache[name] = pygame.transform.scale(surface, (size, size))
        return self._cache[name]

    def _fallback(self, name: str) -> pygame.Surface:
        s = pygame.Surface((32, 32), pygame.SRCALPHA)
        key = name.split('/')[-1]
        colors = {'corridor': (45, 32, 26), 'corridor_cracked': (65, 43, 30),
                  'coal': (28, 28, 30), 'silver': (185, 190, 197), 'gold': (225, 176, 40),
                  'crystal': (45, 195, 185), 'water': (35, 90, 165),
                  'rubble': (100, 93, 88), 'surface': (58, 112, 128),
                  'hud_frame': (49, 42, 37), 'oil': (210, 153, 45),
                  'stability': (90, 160, 85), 'sack': (140, 98, 54)}
        if key.startswith('miner'):
            pygame.draw.rect(s, (150, 100, 48), (10, 12, 13, 18))
            pygame.draw.rect(s, (231, 177, 46), (8, 6, 17, 8))
            lamp = (9, 10) if 'left' in key else (17, 6) if 'up' in key else (17, 14) if 'down' in key else (25, 10)
            pygame.draw.circle(s, (255, 249, 185), lamp, 3)
        elif key == 'prop':
            pygame.draw.rect(s, (134, 87, 47), (13, 0, 6, 32))
            pygame.draw.rect(s, (157, 108, 60), (2, 1, 28, 5))
        elif key == 'exit':
            s.fill((45, 32, 26))
            for x in (9, 23):
                pygame.draw.line(s, (213, 194, 145), (x, 0), (x, 32), 2)
            for y in range(3, 32, 7):
                pygame.draw.line(s, (213, 194, 145), (9, y), (23, y), 2)
        elif key.startswith(('dust', 'collapse')):
            for i in range(9):
                pygame.draw.circle(s, (145, 130, 110, 150), ((i*13)%32, (i*7)%32), 3)
        elif key == 'flash':
            pygame.draw.circle(s, (255, 230, 125, 170), (16, 16), 13)
        else:
            s.fill(colors.get(key, (87, 77, 68)))
            for i in range(8):
                pygame.draw.rect(s, (120, 113, 103), ((i*13)%30, (i*7)%30, 2, 2))
            if key == 'corridor_cracked':
                pygame.draw.lines(s, (20, 15, 12), False, [(4, 0), (16, 12), (12, 22), (22, 32)], 2)
        return s
