# Grafiki kopalni

Grafiki wygenerowano 2026-10-21 narzędziem AI `image_gen.imagegen`
wywołanym przez Codex CLI. Dokładny wariant modelu generatora nie jest
udostępniony. Przyjęte prompty i kolejność komórek zawiera `assets/sheets.json`;
pełny rejestr prób znajduje się w `assets/_raw/prompts_log.txt`.

## Odtwarzanie tekstur

Z katalogu głównego repozytorium, mając Pillow i numpy:

```bash
python scripts/gen_textures.py
```

Surowe arkusze przechowuje się lokalnie w `assets/_raw/` (ignorowane przez Git,
reguła w `.gitignore`). Nie są potrzebne do uruchomienia gry.
Ponowne cięcie wymaga przywrócenia tych arkuszy z kopii; sam manifest nie
odtwarza obrazów. Ponowna generacja z tych samych promptów może dać inne obrazy.
Cięcie tych samych arkuszy w tym samym środowisku jest deterministyczne.

Skrypt usuwa 3% marginesu komórki, usuwa magentę z oznaczonych komórek,
przycina obiekty, dopełnia do kwadratu i skaluje do 32×32. Postacie i stempel
są wyrównane do dołu; pozostałe obiekty do środka. Wszystkie wyniki współdzielą
paletę maksymalnie 48 kolorów RGB, mają kanał alfa 0/255 i format PNG RGBA.
Podgląd z podpisami i szachownicą: `assets/_raw/preview.png`.
Parametry można zmieniać przez `--raw`, `--manifest`, `--out`, `--size`,
`--colors` i `--preview`.

## Kontrakt nazw

Każde `cells[].name` z manifestu oznacza ścieżkę `assets/<name>.png`.
Nazwy odpowiadają `ASSET_NAMES` w `miner_rl/render/assets.py`:
`tiles/` zawiera skały, rudy i elementy kopalni, `sprites/` animacje górnika
oraz efektów, a `ui/` worek, lampę, stabilność i ramkę HUD.
Nie należy zmieniać nazw bez uzgodnienia kontraktu renderera.
Gra działa także bez tekstur: `AssetPack` rysuje wtedy proceduralne zastępniki.

Grafiki są materiałem wygenerowanym przez AI; kontrola techniczna agenta
nie stanowi samodzielnej weryfikacji studenta. Ocena wyglądu w grze
oraz przydatności animacji pozostaje do jego sprawdzenia.
