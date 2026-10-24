# Uczenie się ze wzmocnieniem - projekt 2026/2027

Repozytorium projektu z przedmiotu „Uczenie się ze wzmocnieniem”, WEiI PRz.

**Stan na 24.10.2026:** etap E1 zamknięty (temat „Górnik w niestabilnej kopalni”
przekazany prowadzącemu). Raport zawiera opis problemu, formalizację MDP, opis
środowiska, agentów bazowych z wynikami, opis algorytmów, plan eksperymentów
i wyniki eksperymentu pilotażowego. Gotowe są: środowisko gry (logika, generator
map, grafika, testy), agenci bazowi, Q-learning i SARSA (`miner_rl/agents/tabular.py`)
oraz skrypt treningu (`scripts/train.py`). Pilotaż (4 mapy, 3 ziarna) pokazał, że
agenci uczą się dopiero z optymistyczną inicjalizacją Q i na razie ustępują
heurystyce na trzech z czterech map. Eksperymenty właściwe (etap E3) nie zostały
jeszcze przeprowadzone.

Repozytorium: https://github.com/Kesacky/uczenie-ze-wzmocnieniem

## Dokumentacja

| Ścieżka | Zawartość |
| --- | --- |
| `docs/wymagania.md` | Wymagania z obu dokumentów kursowych i kryteria oceny |
| `docs/etapy.md` | Lista kontrolna E1-E4 i terminy względem bloków P2/P4/P6/P7 |
| `report/main.tex` | Uzupełniany na bieżąco raport projektu |
| `report/sections/ai.tex` | Jawny opis wykorzystania AI i status weryfikacji |
| `scripts/log_ai.py` | Zapis promptów i odpowiedzi wraz z metadanymi |
| `scripts/README.md` | Instrukcja dziennika rozmów z AI |
| `ai-journal/` | Dziennik AI (LaTeX): pełne prompty, odpowiedzi, daty i modele |
| `miner_rl/` | Środowisko (`env/`), grafika pygame (`render/`), agenci (`agents/`), konfiguracja |
| `maps/` | Stałe mapy kopalni 12×11 (stabilna, krucha, kusząca głębia, zawał korytarza) |
| `assets/` | Tekstury pixel art (wygenerowane przez AI) i manifest arkuszy |
| `results/` | Wyniki agentów bazowych (CSV), treningu (`train/`) i nagrania epizodów (`gifs/`) |
| `tests/` | Testy pytest |

Źródłem wymagań są „Zasady realizacji projektu” i „Lista kontrolna projektu”,
2026/2027. Oryginały materiałów kursowych są przechowywane lokalnie; nie są
publikowane bez potwierdzenia prawa do ich udostępniania.

## Uruchomienie gry

Gra „Górnik w niestabilnej kopalni” to środowisko RL z API Gymnasium
(`miner_rl/env/mine_env.py`) z grafiką w pygame. Zasady (stan, akcje, nagrody)
opisuje sekcja „Formalizacja RL/MDP” w raporcie; wszystkie liczby są w
`miner_rl/config.py`. Kopalnie mają 12×11 pól (szyb, poziomy wydobywcze, łączniki i przodki).

**Najprościej (Windows):** dwuklik na `graj.bat` w katalogu projektu. Za pierwszym
razem tworzy środowisko `.venv` i instaluje biblioteki (wymagany Python 3.10+),
potem od razu otwiera menu gry.

W menu wybiera się wariant gry:
- **stała mapa** – jedna z czterech kopalni z `maps/` (stabilna, krucha, kusząca głębia,
  zawał korytarza),
- **losowa mapa** – każda gra na nowej kopalni z generatora (`miner_rl/env/generator.py`).

Każde uruchomienie i każda gra losuje przebieg od nowa (zawały, a w trybie losowym także
kopalnię). Opcja `--seed N` daje powtarzalne gry.

Sterowanie: strzałki to ruch (wejście w złoże = wydobycie), `P` podepnij stempel.
Po końcu gry ekran z wynikiem i rekordem: `R` zagraj ponownie, `N` nowa mapa
(tryb losowy), `M` menu, `Esc` wyjście (przyciski działają też myszą).

Ręcznie (dowolny system):

```powershell
python -m pip install -r requirements.txt
python -m pytest -q                         # testy środowiska, generatora, agentów i grafiki
python scripts/play.py                      # menu
python scripts/play.py --map m3_gleboka     # od razu stała mapa
python scripts/play.py --random --seed 7    # od razu losowa mapa
```

Podgląd agenta (`Spacja` pauza, `+`/`-` tempo). Agent: `random`, `threshold`
(z `--threshold X`), plik modelu `.pkl` zapisany przez `TabularAgent.save` albo `moduł:funkcja`.
Opcja `--random` przełącza na losowe mapy, `--obs-mode local` dodaje do stanu widok otoczenia
górnika (potrzebny agentom uczonym na losowych mapach):

```powershell
python scripts/watch.py --map m3_gleboka --agent threshold --threshold 6 --fps 8
python scripts/watch.py --random --agent threshold --episodes 5
```

Nagranie epizodów do GIF (bez okna; `--only-failed` zapisuje tylko porażki):

```powershell
python scripts/record.py --map m4_zawal_korytarza --agent threshold --threshold 15 --episodes 3 --out results/gifs/
```

Generowanie map (stałe mapy projektu są wygenerowane tym skryptem z zapisanymi ziarnami):

```powershell
python scripts/gen_maps.py --theme losowa --seed 7 --show
python scripts/gen_maps.py --theme gleboka --seed 11 --out maps/moja_mapa.txt
python scripts/gen_maps.py --fixed          # odtwarza maps/m1..m4
```

Ewaluacja agentów bazowych na czterech stałych mapach i na mapach losowych (dobór
progu na ziarnach 0..999, ewaluacja na ziarnach od 1 000 000; wyniki w
`results/baseline_*.csv` i `results/baseline_summary.csv`):

```powershell
python scripts/run_baselines.py --episodes 1000
```

Trening Q-learningu i SARSA (`scripts/train.py`; uruchomienia równolegle na kilku
procesach, wyniki, krzywe uczenia, konfiguracja i modele `.pkl` w `results/train/<nazwa>/`),
a potem wykres krzywych uczenia:

```powershell
python scripts/train.py --maps m1_stabilna m2_krucha m3_gleboka m4_zawal_korytarza --seeds 3 --episodes 50000 --q-init 0 10 20 --name pilot
python scripts/plot_training.py --name pilot
python scripts/watch.py --map m1_stabilna --agent results/train/pilot/models/m1_stabilna_sarsa_q10_s0.pkl
```

Ziarna: trening uruchomienia `s` używa ziaren `s·100 000 + i`, walidacja w trakcie
treningu ziaren od 2 000 000, a ewaluacja końcowa tych samych ziaren od 1 000 000
co agenci bazowi.

Wspólne opcje grafiki: `--scale auto|1|1.5|2` (domyślnie dopasowanie do ekranu),
`--no-textures` (kolorowe zastępniki zamiast tekstur), `--no-lamp` (bez ciemności poza
światłem lampy). Bez okna, np. na serwerze: zmienna środowiskowa `SDL_VIDEODRIVER=dummy`.

Tekstury w `assets/` wygenerowano narzędziem AI (szczegóły i odtwarzanie:
`assets/README.md`, `python scripts/gen_textures.py`). Gra działa także bez nich.

## Uruchomienie dokumentacji

Instrukcja gry, agentów bazowych i treningu jest w sekcji „Uruchomienie gry”.

Do narzędzia dziennika wystarczy Python 3.10 lub nowszy, bez dodatkowych bibliotek:

```powershell
python scripts/log_ai.py --help
```

Raport wymaga dystrybucji LaTeX z obsługą języka polskiego (pdfLaTeX lub XeLaTeX).
Dziennik AI (`ai-journal/main.tex`) kompiluje się XeLaTeX-em (`xelatex main.tex`).
Raport:

```powershell
Set-Location report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

W Codex można także otworzyć katalog `report` w edytorze LaTeX i skompilować
go wbudowanym kompilatorem. Na etapie przygotowania nie jest wymagany trening
ani instalacja bibliotek uczenia maszynowego.

## Dokumentowanie AI

P00, sekcja 9, wymaga sekcji „Wsparcie AI” w raporcie oraz zapisanych transkryptów
rozmów z datą i nazwą modelu. Sekcja w raporcie nie zastępuje pełnego transkryptu.
Należy odnotować cel i zakres pomocy oraz wskazać, co student sam zweryfikował
lub zmodyfikował. Testy wykonane przez AI nie są weryfikacją studenta.

Pełny dziennik LaTeX i oryginalne teksty wymian znajdują się w repozytorium
w katalogu `ai-journal/` (dokument główny: `ai-journal/main.tex`), więc prowadzący
ma do nich bezpośredni dostęp. Rozmowy spoza narzędzi programistycznych należy
dodatkowo udostępnić przez link, o ile narzędzie daje taką możliwość.

Skrypt zapisuje dostarczone teksty; nie przechwytuje automatycznie rozmów.
Każda nowa rozmowa wymaga zapisania jej pełnego przebiegu i aktualizacji raportu.

## Najbliższy krok

Etap E3: eksperymenty właściwe zgodnie z planem w raporcie (10 ziaren, `q_init` = 10,
wpływ kary za zawał, stała a losowa mapa) oraz wyjaśnienie, dlaczego agenci RL
wracają przed wydobyciem wszystkich złóż. Daty kalendarzowe bloków zajęć należy
wpisać po ich potwierdzeniu.
