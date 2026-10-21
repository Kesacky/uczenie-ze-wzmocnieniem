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

## 3. Agenci i eksperymenty

- **Baselines:** agent losowy oraz heurystyka „kop do progu ładunku X, potem wracaj
  najkrótszą drogą” (kilka wartości X, wybrana najlepsza, żeby porównanie było uczciwe).
- **RL:** Q-learning i SARSA (porównanie podejścia do ryzyka), opcjonalnie Double Q-learning.
  Rozszerzenie: DQN na większych mapach.
- **Mapy:** 3–4 kopalnie o rosnącej trudności (stabilna, krucha, „kusząca głębia”, z zawałem korytarza).
- **Eksperymenty:** wpływ γ i kary za zawał na apetyt na ryzyko, porównanie α i ε,
  kilka ziaren losowych na wariant, krzywe uczenia, rozdzielenie treningu od ewaluacji
  oraz analiza nieudanych epizodów (powtórki jako GIF).

## 4. Gra (implementacja z Codexem)

- **Architektura:** `miner_rl/env` to czysta logika, API Gymnasium (`reset`, `step`), bez grafiki,
  szybka do treningu. `miner_rl/render` to grafika w pygame. `miner_rl/agents` to baseline'y,
  Q-learning, SARSA i (opcjonalnie) DQN. `scripts/` zawiera trening, ewaluację i nagrywanie.
- **Tryby:** `play` (człowiek, strzałki i klawisze akcji), `watch` (wytrenowany agent,
  regulacja prędkości) oraz `record` (GIF lub MP4 do README i raportu).
- **HUD:** stabilność stropu (pasek, przy niskim poziomie drży i sypie się kurz), olej w lampie,
  ładunek i wynik. Ciemność poza promieniem lampy (opcjonalnie).
- **Grafika:** tekstury i sprite'y generowane przez Codexa (funkcja image_generation).
  Spójny styl pixel art, kafelki 32×32 lub 48×48: skała, korytarz, złoża (np. węgiel,
  srebro, złoto, kryształ), stempel, wyjście, górnik (4 kierunki i kopanie), animacja zawału.
  Zasady: zapisywać prompty do grafik w dzienniku AI i opisać je w sekcji „Wsparcie AI”.
  Grafika ma zapasowy wariant z kolorowymi prostokątami, żeby trening i testy nie zależały
  od plików graficznych.
- **Testy:** reset/step, granice mapy, niedozwolone akcje, zawał, brak oleju, wyjście z ładunkiem,
  limit kroków, powtarzalność przy ustalonym ziarnie.

## 5. Kolejność prac

1. E1: opis problemu i formalizacja w raporcie, wysłanie tematu do akceptacji.
2. E2: środowisko i testy, baseline'y z zapisanymi wynikami, plan eksperymentów, wersja graficzna.
3. E3: Q-learning i SARSA (opcjonalnie DQN), eksperymenty, wykresy, analiza, raport.
4. E4: poprawki po uwagach prowadzącego.
