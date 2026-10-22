# Koncepcja projektu: Górnik w niestabilnej kopalni

Status: **propozycja wstępna** (21.10.2026). Temat nie jest jeszcze zaakceptowany przez prowadzącego.
Termin akceptacji: najpóźniej do końca P2 (etap E1). Ścieżka: poziom podstawowy, środowisko gridowe
(P00, sekcja 6.1), z możliwym rozszerzeniem o DQN.

## 1. Problem

Górnik schodzi do kopalni po złoża. Głębsze złoża są cenniejsze, ale każde kopanie osłabia
strop i zwiększa ryzyko zawału. Lampa ma ograniczoną ilość oleju. Agent sam decyduje, kiedy
kopać dalej, a kiedy wrócić do wyjścia z tym, co zebrał. Nagrodę dostaje dopiero za ładunek
wyniesiony na powierzchnię. Przy zawale traci ładunek.

**Dlaczego RL:** decyzje są sekwencyjne i zmieniają przyszły stan kopalni (stabilność, złoża,
olej). Wynik jest losowy (zawał), a nagroda opóźniona. Trzeba ocenić ryzyko, a nie tylko znaleźć
najkrótszą drogę, więc zwykłe przeszukiwanie grafu nie wystarcza. Pytanie „kopać jeszcze czy
wracać?” to klasyczny kompromis między zyskiem a ryzykiem.

## 2. Formalizacja MDP

Aktualna, szczegółowa wersja jest w raporcie (`report/main.tex`, sekcja „Formalizacja RL/MDP”).
Decyzje z 21.10.2026: stabilność w 6 poziomach (0–5), globalna dla całej kopalni; stała kara −20
za zawał (ładunek przepada); 5 akcji (4 ruchy i „podepnij”, wejście w złoże to wydobycie), bez akcji „czekaj”;
ładunek wynika z maski wydobytych złóż, więc nie jest osobnym elementem stanu.

Po pierwszych testach i zagraniu w grę (21.10.2026, wieczór): mapy 12×11 o układzie kopalni
(szyb, poziomy, łączniki), ryzyko zawału p = 0,001 + 0,003·(5−σ)²·(1+y/H) (niewielkie ryzyko także
przy pełnej stabilności), wartości złóż 3/9/18/30. Wszystkie liczby są w `miner_rl/config.py`.

## 3. Warianty gry

- **Stała mapa:** jedna z czterech kopalni (`maps/`: stabilna, krucha, kusząca głębia, zawał korytarza).
- **Losowa mapa:** w każdym epizodzie nowa kopalnia z generatora (`miner_rl/env/generator.py`);
  obserwacja agenta jest wtedy rozszerzona o widok otoczenia 5×5.

## 4. Agenci i eksperymenty

- **Agenci bazowi:** agent losowy oraz heurystyka progowa (próg ładunku X i moment stawiania
  stempla dobierane osobno dla każdej mapy). Wyniki: `results/`, tabela w raporcie.
- **RL:** Q-learning i SARSA (`miner_rl/agents/tabular.py`), 10 ziaren na wariant.
- **Eksperymenty:** porównanie główne na 4 mapach i mapach losowych; wpływ kary za zawał
  (−5/−20/−50); agent uczony na stałej mapie a agent uczony na losowych mapach (uogólnianie
  na nowe kopalnie). Rozszerzenie, jeśli zostanie czas: DQN na mapach losowych.

## 5. Gra (zrealizowana)

Logika w `miner_rl/env` (bez grafiki), grafika pygame w `miner_rl/render`, tekstury pixel art
wygenerowane przez Codexa (`assets/`), tryby: gra (`scripts/play.py`, `graj.bat`), podgląd agenta
(`scripts/watch.py`) i nagrywanie GIF-ów (`scripts/record.py`). Testy: `python -m pytest -q`.
