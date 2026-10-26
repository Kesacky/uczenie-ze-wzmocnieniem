# Uczenie się ze wzmocnieniem - projekt 2026/2027

Repozytorium projektu z przedmiotu „Uczenie się ze wzmocnieniem”, WEiI PRz.

**Stan na 24.10.2026:** etap E1 zamknięty (temat „Górnik w niestabilnej kopalni”
przekazany prowadzącemu), E2 wykonany. Raport zawiera opis problemu, formalizację
MDP, opis środowiska, agentów bazowych z wynikami, opis algorytmów, plan eksperymentów,
wyniki pilotażu oraz doboru hiperparametrów i diagnozy zachowania agenta (początek E3). Gotowe są: środowisko gry (logika, generator
map, grafika, testy), agenci bazowi, Q-learning i SARSA (`miner_rl/agents/tabular.py`)
oraz skrypty treningu i analizy. Od 24.10.2026 obowiązują zasady w wersji 3:
kara za akcję niedozwoloną −2, za zawał −10, nagroda 0,3 · wartość złoża za
wydobycie, a agenci maskują akcje niedozwolone. Od 25.10.2026 trening używa
malejącego α (0,1 → 0,01) i kary za krok −0,25 (ocena nadal według zasad gry).
Usunęło to nieudane uruchomienia i zapętlenia agenta. W konfiguracji końcowej
(10 ziaren, 100 tys. epizodów) agenci RL wygrywają z heurystyką na mapie kruchej
(+32 pkt) i głębokiej (Q-learning, +5 pkt), a przegrywają na stabilnej (−6) i zawału
korytarza (−14). Eksperyment 2 (kara za zawał −5 / −10 / −20) pokazał tylko niewielki
spadek odsetka zawałów przy wyższej karze, bez płytszego kopania. Eksperyment 3:
agent tablicowy nie uogólnia na nowe kopalnie — uczony na mapach losowych zdobywa
ok. −17 pkt (heurystyka 61,7), a uczony na mapie stałej na innej mapie ok. −3 pkt.
Dodatkowy eksperyment 4 (pomysł studenta): obserwacja z podpowiedzią „narratora” (kierunek
i odległość do najbliższego złoża i do wyjścia zamiast widoku otoczenia). Z kierunkiem
pierwszego kroku najkrótszej drogi Q-learning na nowych kopalniach zdobywa 30,8 pkt,
z samym kierunkiem „na przełaj” 5,2 pkt, a z dodatkową pamięcią odwiedzonych pól 37,6 pkt
(agent nadal często krąży między polami).

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
górnika (potrzebny agentom uczonym na losowych mapach), a `--obs-mode hint --hint-kind compass|path`
daje obserwację z podpowiedzią narratora (modele `ho1` i `ho2` z eksperymentu 4):

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
python scripts/train.py --maps m1_stabilna m2_krucha m3_gleboka m4_zawal_korytarza --seeds 10 --episodes 100000 --name koncowy
python scripts/plot_training.py --name koncowy
python scripts/watch.py --map m3_gleboka --agent results/train/koncowy/models/m3_gleboka_q_learning_q30_n100000_s0.pkl
```

Eksperymenty 2 i 3 (kara za zawał; mapa stała a losowa z oceną krzyżową `scripts/cross_eval.py`).
Trening na mapach losowych zajmuje ok. 1,5–1,8 GB RAM na proces, stąd mniej procesów:

```powershell
python scripts/train.py --maps m1_stabilna m2_krucha m3_gleboka m4_zawal_korytarza --seeds 10 --episodes 100000 --reward-collapse -5 -20 --name kara_zawal
python scripts/train.py --maps m1_stabilna m2_krucha m3_gleboka m4_zawal_korytarza --seeds 10 --episodes 100000 --local-obs 1 --name mapy_stale_lokalne
python scripts/train.py --maps losowa --seeds 10 --episodes 100000 --local-obs 1 --workers 3 --name mapy_losowe
python scripts/cross_eval.py --runs mapy_stale_lokalne mapy_losowe --targets m1_stabilna m2_krucha m3_gleboka m4_zawal_korytarza losowa --variants q30_n100000_lo1 --workers 4 --out results/train/mapy
```

Eksperyment 4 (dodatkowy): obserwacja z podpowiedzią narratora (`--hint-obs 1` kompas,
`--hint-obs 2` droga). Tablice są małe (ok. 0,1 GB na proces):

```powershell
python scripts/train.py --maps losowa --seeds 10 --episodes 100000 --hint-obs 1 2 --workers 8 --name narrator_losowe
python scripts/train.py --maps m1_stabilna m2_krucha m3_gleboka m4_zawal_korytarza --seeds 10 --episodes 100000 --hint-obs 2 --workers 8 --name narrator_stale
python scripts/cross_eval.py --runs narrator_stale narrator_losowe --targets m1_stabilna m2_krucha m3_gleboka m4_zawal_korytarza losowa --variants q30_n100000_ho1 q30_n100000_ho2 --workers 6 --out results/train/narrator
python scripts/plot_training.py --name narrator_losowe --variants q30_n100000_ho1 q30_n100000_ho2
python scripts/watch.py --random --obs-mode hint --hint-kind path --agent results/train/narrator_losowe/models/losowa_q_learning_q30_n100000_ho2_s0.pkl
python scripts/train.py --maps losowa --seeds 10 --episodes 100000 --hint-obs 3 --workers 8 --name narrator_pamiec
python scripts/watch.py --random --obs-mode hint --hint-kind path --hint-memory --agent results/train/narrator_pamiec/models/losowa_q_learning_q30_n100000_ho3_s8.pkl
```

Przegląd hiperparametrów: każdy z `--q-init`, `--alpha`, `--gamma`, `--eps-decay`,
`--eps-end`, `--episodes`, `--oil-bins` przyjmuje kilka wartości (siatka kombinacji,
a z `--ofat` zmiana jednego czynnika naraz). Analiza zachowania agenta (które złoża
wydobywa, akcje niedozwolone, mapa cieplna odwiedzin) i wykres wpływu Q0:

```powershell
python scripts/train.py --ofat --maps m1_stabilna --q-init 10 30 60 --alpha 0.1 0.3 0.5 --name przeglad_m1
python scripts/analyze_policy.py --map m1_stabilna --agent results/train/q0_mapy/models/m1_stabilna_sarsa_q30_s0.pkl --heatmap mapa.png
python scripts/plot_q0.py --runs przeglad_m1 q0_m1 olej_m1 --map m1_stabilna
```

Agenta uczonego z `--oil-bins N` oglądamy i analizujemy z tą samą opcją `--oil-bins N`.

Zmiany nagrody w treningu (`--reward-illegal`, `--reward-collapse`, `--dig-fraction`,
`--dig-refund`) i maskowanie akcji niedozwolonych (`--mask-illegal 1`); ewaluacja zawsze
używa domyślnych nagród gry. Zestawienie wariantów na wszystkich mapach:

```powershell
python scripts/train.py --maps m1_stabilna m2_krucha m3_gleboka m4_zawal_korytarza --q-init 30 --reward-illegal -2 --reward-collapse -10 --dig-fraction 0.3 --mask-illegal 0 1 --name nagrody_c
python scripts/compare_runs.py --runs nagrody_a nagrody_b nagrody_c --out results/train/nagrody
```

Domyślne parametry `train.py` to konfiguracja końcowa (Q0 = 30, α malejące do 0,01, kara za krok
−0,25 w treningu, maskowanie akcji). Ewaluacja podaje też politykę najlepszą według walidacji
(wariant `_best`) i odsetek epizodów z pętlą (`loop_rate`). Wyniki: `results/train/kara_krok/`
(konfiguracja końcowa to wariant `q30_ae0.01_n100000_rs-0.25`), wcześniej `stabilizacja/` i `glowny/`.
Przerwany trening można wznowić tym samym poleceniem: gotowe uruchomienia są zapisywane
na bieżąco w `partial/` i pomijane. Wyniki w `results/train/` sprzed 24.10.2026
(`pilot`, `przeglad_m1`, `q0_m1`, `olej_m1`, `q0_mapy`, `nagrody_*`) oceniano według
zasad w wersji 2 (parametry w ich `config.json`, heurystyka: `results/baseline_summary_v2.csv`).
Premia za eksplorację: `--bonus β`; kształtowanie potencjałowe: `--dig-refund 1`.

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

Etap E3: analiza i wnioski w raporcie (do napisania przez studenta). Ewentualnie
DQN dla map losowych, jeśli student zdecyduje się na tę próbę. Daty kalendarzowe bloków zajęć należy
wpisać po ich potwierdzeniu.
