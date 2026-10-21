#!/usr/bin/env python3
"""Tnie arkusze AI na tekstury RGBA ze wspólną paletą."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def prepare(cell, key, name, size):
    """Usuwa margines i tło, skaluje oraz ustala binarną alfę."""
    w, h = cell.size
    mx, my = round(w * .03), round(h * .03)
    cell = cell.crop((mx, my, w - mx, h - my)).convert('RGBA')
    if key == 'none':
        cell = cell.resize((size, size), Image.Resampling.BOX)
        cell.putalpha(255)
        return cell
    if key != 'magenta':
        raise ValueError(f'Nieznany klucz tła: {key}')
    data = np.array(cell)
    rgb = data[:, :, :3].astype(np.int32)
    r, g, b = rgb.transpose(2, 0, 1)
    mask = (np.sum((rgb - [255, 0, 255]) ** 2, axis=2) < 110 ** 2)
    mask |= (r > 120) & (b > 100) & (g < np.minimum(r, b) * .65)
    data[mask] = 0
    cell = Image.fromarray(data)
    box = cell.getbbox()
    if box is None:
        print(f'Ostrzeżenie: pusta komórka {name}')
        return Image.new('RGBA', (size, size))
    if np.count_nonzero(data[:, :, 3]) < data.shape[0] * data.shape[1] * .01:
        print(f'Ostrzeżenie: prawie pusta komórka {name}')
    cell = cell.crop(box)
    edge = max(cell.size)
    square = Image.new('RGBA', (edge, edge))
    bottom = name.startswith('sprites/miner_') or name == 'tiles/prop'
    square.paste(cell, ((edge - cell.width) // 2,
                       edge - cell.height if bottom else (edge - cell.height) // 2))
    square = square.resize((size, size), Image.Resampling.BOX)
    data = np.array(square)
    data[:, :, 3] = np.where(data[:, :, 3] >= 128, 255, 0)
    data[data[:, :, 3] == 0] = 0
    return Image.fromarray(data)


def preview(tiles, path, size):
    """Rysuje podgląd na szachownicy z podpisami."""
    width, height = max(180, size * 4 + 8), size * 4 + 36
    cols = 5
    board = Image.new('RGB', (cols * width, ((len(tiles) + cols - 1) // cols) * height), '#242424')
    draw = ImageDraw.Draw(board)
    for i, (name, tile) in enumerate(tiles):
        x, y = (i % cols) * width, (i // cols) * height
        for yy in range(0, size * 4, 8):
            for xx in range(0, size * 4, 8):
                color = '#666666' if (xx // 8 + yy // 8) % 2 else '#999999'
                draw.rectangle((x + xx, y + yy, x + xx + 7, y + yy + 7), fill=color)
        large = tile.resize((size * 4, size * 4), Image.Resampling.NEAREST)
        board.paste(large, (x, y), large)
        draw.text((x + 2, y + size * 4 + 4), name, fill='white')
    path.parent.mkdir(parents=True, exist_ok=True)
    board.save(path)
    print(f'Zapisano podgląd: {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('assets/_raw'))
    parser.add_argument('--manifest', type=Path, default=Path('assets/sheets.json'))
    parser.add_argument('--out', type=Path, default=Path('assets'))
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--colors', type=int, default=48)
    parser.add_argument('--preview', type=Path, default=Path('assets/_raw/preview.png'))
    args = parser.parse_args(argv)
    if args.size < 1 or not 1 <= args.colors <= 256:
        parser.error('Rozmiar musi być dodatni, liczba kolorów od 1 do 256.')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    sheets = manifest if isinstance(manifest, list) else manifest['sheets']
    tiles = []
    seen = set()
    for sheet in sheets:
        cols, rows = sheet['grid']
        if cols < 1 or rows < 1 or len(sheet['cells']) != cols * rows:
            raise ValueError('Niepoprawna siatka lub liczba komórek.')
        with Image.open(args.raw / sheet['file']) as image:
            if image.width * rows != image.height * cols:
                raise ValueError(f"Komórki arkusza {sheet['file']} nie są kwadratowe.")
            for i, spec in enumerate(sheet['cells']):
                name = spec['name']
                if Path(name).is_absolute() or '..' in Path(name).parts or name in seen:
                    raise ValueError(f'Niepoprawna lub powtórzona nazwa: {name}')
                seen.add(name)
                x, y = i % cols, i // cols
                cell = image.crop((x * image.width // cols, y * image.height // rows,
                                   (x + 1) * image.width // cols, (y + 1) * image.height // rows))
                tiles.append((name, prepare(cell, spec['key'], name, args.size)))
    pixels = [np.array(tile)[:, :, :3][np.array(tile)[:, :, 3] > 0] for _, tile in tiles]
    pixels = np.concatenate(pixels) if pixels else np.empty((0, 3), dtype=np.uint8)
    if not len(pixels):
        raise ValueError('Brak nieprzezroczystych pikseli do budowy palety.')
    # Każdy odcień ma równą wagę: dominująca skała nie wypiera zieleni
    # trawy i błękitów rud występujących tylko na kilku kafelkach.
    pixels = np.unique(pixels, axis=0)
    hsv = np.array(Image.fromarray(pixels.reshape(1, -1, 3)).convert('HSV'))[0]
    groups = np.where(hsv[:, 1] < 32, 16, hsv[:, 0] // 16)
    # Równoważymy rodziny barw, nadal używając wyłącznie kolorów wejścia.
    samples = []
    for group in np.unique(groups):
        values = pixels[groups == group]
        samples.append(values[np.linspace(0, len(values) - 1, 1024, dtype=int)])
    pixels = np.concatenate(samples)
    palette = Image.fromarray(pixels.reshape(1, -1, 3)).quantize(
        colors=args.colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    # Wypełnienie nieużywanych indeksów zapobiega dodaniu czerni spoza palety.
    colors = palette.getpalette()[:args.colors * 3]
    palette.putpalette(colors + colors[:3] * (256 - args.colors))
    results = []
    for name, tile in tiles:
        alpha = tile.getchannel('A')
        result = tile.convert('RGB').quantize(palette=palette, dither=Image.Dither.NONE).convert('RGBA')
        result.putalpha(alpha)
        destination = args.out / (name + '.png')
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.save(destination)
        print(f'Zapisano: {destination}')
        results.append((name, result))
    if args.preview:
        preview(results, args.preview, args.size)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
