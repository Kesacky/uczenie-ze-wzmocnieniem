#!/usr/bin/env python3
"""Generowanie map kopalni do plików `maps/*.txt` (algorytm: miner_rl/env/generator.py).

    python scripts/gen_maps.py --fixed                 # odtwarza cztery stałe mapy projektu
    python scripts/gen_maps.py --theme losowa --seed 7 --out maps/moja.txt
    python scripts/gen_maps.py --theme gleboka --seed 3 --show   # tylko podgląd w konsoli

Stałe mapy projektu są wygenerowane tym skryptem z ustalonymi ziarnami (FIXED niżej),
więc każdą można odtworzyć. Tryb gry z losową mapą (`play.py --random`,
`MineEnv(random_maps=True)`) używa motywu `losowa` z ziarnem epizodu.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from miner_rl.env.generator import GeneratorConfig, generate_map, map_to_text  # noqa: E402

THEMES: dict[str, tuple[str, GeneratorConfig]] = {
    "losowa": ("Losowa kopalnia", GeneratorConfig()),
    "stabilna": ("Stabilna kopalnia: płytkie, tańsze złoża",
                 GeneratorConfig(depth_bands=((0, .3), (0, .4), (.1, .5), (.2, .6), (.3, .7)),
                                 collapse_prob=0.0)),
    "krucha": ("Krucha kopalnia: niska stabilność na starcie i jeden stempel",
               GeneratorConfig(collapse_prob=0.0, overrides={"stability_start": 3, "props_start": 1})),
    "gleboka": ("Kusząca głębia: tanie złoża płytko, kryształy na samym dnie",
                GeneratorConfig(depth_bands=((0, .15), (0, .2), (.75, 1), (.85, 1), (.85, 1)),
                                collapse_prob=0.0)),
    "zawal": ("Zawał korytarza: po wydobyciu 3 złóż pola X zamieniają się w skałę",
              GeneratorConfig(collapse_prob=1.0, min_detour=4)),
}
FIXED = {   # plik -> (motyw, ziarno)
    "m1_stabilna": ("stabilna", 1),
    "m2_krucha": ("krucha", 2),
    "m3_gleboka": ("gleboka", 3),
    "m4_zawal_korytarza": ("zawal", 2),
}


def make(theme: str, seed: int, name: str):
    description, gen = THEMES[theme]
    mine_map = generate_map(np.random.default_rng(seed), gen, name=name)
    header = {"opis": description,
              "wygenerowano": f"python scripts/gen_maps.py --theme {theme} --seed {seed}"}
    return mine_map, map_to_text(mine_map, header)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixed", action="store_true", help="zapisz cztery stałe mapy do maps/")
    parser.add_argument("--theme", choices=sorted(THEMES), default="losowa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="plik wyjściowy (bez --out: tylko podgląd)")
    parser.add_argument("--show", action="store_true", help="wypisz mapę w konsoli")
    args = parser.parse_args(argv)

    if args.fixed:
        for name, (theme, seed) in FIXED.items():
            mine_map, text = make(theme, seed, name)
            path = ROOT / "maps" / f"{name}.txt"
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"{path.relative_to(ROOT)}: złoża {''.join(ch for *_, ch in mine_map.deposits)}, "
                  f"pola X: {len(mine_map.collapse_cells)}")
        return 0
    name = args.out.stem if args.out else f"{args.theme}_{args.seed}"
    mine_map, text = make(args.theme, args.seed, name)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"Zapisano {args.out}")
    if args.show or not args.out:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
