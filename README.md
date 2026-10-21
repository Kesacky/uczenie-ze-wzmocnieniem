# Uczenie się ze wzmocnieniem - projekt 2026/2027

Repozytorium projektu z przedmiotu „Uczenie się ze wzmocnieniem”, WEiI PRz.

**Stan na 21.10.2026:** przygotowano strukturę dokumentacji, zestawienie wymagań,
szablon raportu LaTeX oraz narzędzie do zapisywania rozmów z AI. Proponowany temat
„Górnik w niestabilnej kopalni” czeka na akceptację prowadzącego. Jest środowisko
gry (logika, mapy, grafika, testy) oraz agenci bazowi z pierwszymi wynikami;
eksperymenty z agentami RL nie zostały jeszcze przeprowadzone. Etap E1 pozostaje otwarty.

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
| `maps/` | Mapy kopalni 8×10 (stabilna, krucha, kusząca głębia, zawał korytarza) |
| `assets/` | Tekstury pixel art (wygenerowane przez AI) i manifest arkuszy |
| `results/` | Wyniki agentów bazowych (CSV) i nagrania epizodów (`gifs/`) |
| `tests/` | Testy pytest |

Źródłem wymagań są „Zasady realizacji projektu” i „Lista kontrolna projektu”,
2026/2027. Oryginały materiałów kursowych są przechowywane lokalnie; nie są
publikowane bez potwierdzenia prawa do ich udostępniania.

## Uruchomienie gry

Gra „Górnik w niestabilnej kopalni” to środowisko RL z API Gymnasium
(`miner_rl/env/mine_env.py`) z grafiką w pygame. Zasady (stan, akcje, nagrody)
opisuje sekcja „Formalizacja RL/MDP” w raporcie; wszystkie liczby są w
`miner_rl/config.py`, a mapy w `maps/*.txt`.

Wymagania: Python 3.10+.

```powershell
python -m pip install -r requirements.txt
python -m pytest -q                      # testy środowiska, agentów bazowych, grafiki i tekstur
```

Gra człowieka (strzałki: ruch, wejście w złoże = wydobycie; `P` podepnij; `R` restart; `Esc` wyjście):

```powershell
python scripts/play.py --map maps/m1_stabilna.txt
```

Podgląd agenta (`Spacja` pauza, `+`/`-` tempo). Agent: `random`, `threshold`
(z `--threshold X`), plik modelu `.pkl` zapisany przez `TabularAgent.save` albo `moduł:funkcja`:

```powershell
python scripts/watch.py --map m3_gleboka --agent threshold --threshold 6 --fps 8
```

Nagranie epizodów do GIF (bez okna; `--only-failed` zapisuje tylko porażki):

```powershell
python scripts/record.py --map m4_zawal_korytarza --agent threshold --threshold 15 --episodes 3 --out results/gifs/
```

Ewaluacja agentów bazowych (dobór progu na ziarnach 0..999, ewaluacja na ziarnach
od 1 000 000; wyniki w `results/baseline_*.csv` i `results/baseline_summary.csv`):

```powershell
python scripts/run_baselines.py --episodes 1000
```

Wspólne opcje grafiki: `--scale 2`, `--no-textures` (kolorowe zastępniki zamiast
tekstur), `--no-lamp` (bez ciemności poza światłem lampy). Bez okna, np. na
serwerze: zmienna środowiskowa `SDL_VIDEODRIVER=dummy`.

Tekstury w `assets/` wygenerowano narzędziem AI (szczegóły i odtwarzanie:
`assets/README.md`, `python scripts/gen_textures.py`). Gra działa także bez nich.

## Uruchomienie dokumentacji

Instrukcja gry i agentów bazowych jest w sekcji „Uruchomienie gry”. Trening
i konfiguracje eksperymentów RL zostaną opisane wraz z ich wykonaniem.

Do narzędzia dziennika wystarczy Python 3.10 lub nowszy, bez dodatkowych bibliotek:

```powershell
python scripts/log_ai.py --help
```

Raport wymaga dystrybucji LaTeX z obsługą języka polskiego (pdfLaTeX):

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

Wybrać temat, opisać cel agenta i wstępne MDP, potwierdzić wymagane uzgodnienia
z prowadzącym oraz przekazać mu link do repozytorium. Daty kalendarzowe bloków
zajęć należy wpisać po ich potwierdzeniu.
