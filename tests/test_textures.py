"""Sprawdzenie cięcia i wspólnej palety bez surowych grafik projektu."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def test_synthetic_sheet(tmp_path):
    spec = importlib.util.spec_from_file_location('gen_textures', Path(__file__).resolve().parents[1] / 'scripts/gen_textures.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = tmp_path / 'raw'
    raw.mkdir()
    sheet = Image.new('RGB', (128, 64), '#ff00ff')
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, 63, 63), fill='#443322')
    draw.rectangle((80, 12, 111, 54), fill='#66bb33')
    draw.rectangle((88, 20, 100, 35), fill='#ffee88')
    sheet.save(raw / 'sheet.png')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'sheets': [{'file': 'sheet.png', 'grid': [2, 1], 'cells': [
        {'name': 'tiles/full', 'key': 'none'}, {'name': 'sprites/miner_test', 'key': 'magenta'}]}]}))
    out = tmp_path / 'out'
    args = ['--raw', str(raw), '--manifest', str(manifest), '--out', str(out),
            '--colors', '2', '--preview', str(tmp_path / 'preview.png')]
    assert module.main(args) == 0
    colors = set()
    snapshots = []
    for name in ('tiles/full', 'sprites/miner_test'):
        with Image.open(out / (name + '.png')) as image:
            assert image.mode == 'RGBA'
            assert image.size == (32, 32)
            data = np.array(image)
            assert set(np.unique(data[:, :, 3])) <= {0, 255}
            colors.update(map(tuple, data[:, :, :3][data[:, :, 3] > 0]))
            snapshots.append(image.tobytes())
            if name.startswith('sprites'):
                assert data[0, 0, 3] == 0
                assert np.any(data[:, :, 3] == 255)
            else:
                assert np.all(data[:, :, 3] == 255)
    assert len(colors) <= 2
    assert (tmp_path / 'preview.png').exists()
    module.main(args)
    for name, expected in zip(('tiles/full', 'sprites/miner_test'), snapshots):
        with Image.open(out / (name + '.png')) as image:
            assert image.tobytes() == expected
