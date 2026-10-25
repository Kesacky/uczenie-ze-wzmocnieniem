# Etapy i lista kontrolna

Stan na 24.10.2026: **E1 zamknięty** według oświadczenia studenta z 24.10.2026 (temat „Górnik w niestabilnej kopalni” i link do repozytorium przekazane prowadzącemu; data i forma przekazania nie zostały zapisane). Prace E2 są w toku: gotowe są środowisko gry, generator map, agenci bazowi z wynikami, Q-learning i SARSA oraz skrypt treningu z pierwszymi wynikami (`results/train/pilot/`).

Publiczne repozytorium: [Kesacky/uczenie-ze-wzmocnieniem](https://github.com/Kesacky/uczenie-ze-wzmocnieniem). Samo umieszczenie linku w dokumentacji nie oznacza przekazania go prowadzącemu ani zaliczenia etapu.

Źródła: P00 = lokalny `P00_zasady_realizacji_projektu.pdf`; P01 = lokalny `P01_checklista_projektu.pdf`. Pełne wymagania i punktacja: [wymagania.md](wymagania.md).

## Harmonogram

| Etap | Termin względny | Data kalendarzowa | Wymagany rezultat | Status |
| --- | --- | --- | --- | --- |
| E1 | do końca P2 | do potwierdzenia | temat, link do repo, opis problemu, wstępna formalizacja | zamknięty (oświadczenie studenta, 24.10.2026) |
| E2 | do P4 | do potwierdzenia | środowisko, baseline, plan badań, pierwsza działająca wersja | wykonany (24.10.2026), czeka na weryfikację prowadzącego |
| E3 | do P6 | do potwierdzenia | finalny projekt, raport, eksperymenty, wyniki i analiza | nierozpoczęty |
| E4 | P7 | do potwierdzenia | poprawki wskazane przez prowadzącego, domknięcie | oczekuje na uwagi |

Źródło: P00 §3–4, s. 2–3. PDF-y nie podają dat kalendarzowych; uzupełnić je na podstawie planu zajęć lub informacji prowadzącego. E4 służy poprawkom, a nie wykonywaniu projektu od początku (P01 §5, s. 5–6).

Każdy rozpoczęty tydzień opóźnienia może kosztować 10 punktów, osobno dla każdego etapu (P00 §5, s. 3). Przed etapem należy sprawdzić repo, raport, uruchamialność i gotowość do wyjaśnienia pracy.

## E1 — temat, repozytorium, formalizacja

Źródła: P00 §2, s. 1–2; §4, s. 3; P01 §2, s. 1–2.

- [x] Przeczytać zasady i checklistę projektu.
- [x] Spisać wymagania i etapy wraz ze źródłami.
- [x] Potwierdzić, że publiczne repo GitHub zostało utworzone i jest dostępne (21.10.2026, GitHub potwierdził widoczność `public`).
- [x] Przekazać link prowadzącemu wskazaną przez niego drogą i zapisać datę/czynność (wykonane według oświadczenia studenta z 24.10.2026; dokładna data i forma: do uzupełnienia).
- [x] Dodać README z prawdziwym opisem aktualnego stanu (`README.md`).
- [x] Dodać szkielet sprawozdania `report/main.tex` i sprawdzić jego kompilację (21.10.2026, kompilator edytora: sukces).
- [x] Wykonać pierwszy commit rzeczywiście utworzonych materiałów (commit inicjujący dokumentację i narzędzie dziennika, 21.10.2026).
- [x] Sprawdzić przed publikacją, czy brak sekretów, danych wrażliwych i materiałów bez prawa rozpowszechniania; oryginalne PDF-y zostawić lokalnie (kontrola plików i `git check-ignore`).
- [x] Wybrać i jednoznacznie opisać temat (raport, sekcja 1). Uzgodnić temat z prowadzącym: zamknięte według oświadczenia studenta z 24.10.2026.
- [x] Zapisać cel agenta i uzasadnienie sekwencyjnego charakteru problemu oraz zastosowania RL (raport, sekcje 1–2).
- [x] Opisać agenta, środowisko i stan/obserwację (raport, sekcja 3).
- [x] Zdefiniować dostępne akcje, nagrody i warunki końca epizodu (raport, sekcja 3).
- [x] Opisać ograniczenia, pojedynczy krok interakcji i mierzalną definicję sukcesu (raport, sekcja 3).
- [x] Wskazać niepewne założenia formalizacji do weryfikacji po testach (raport, sekcja 3, „Elementy do weryfikacji”).
- [x] Udokumentować dotychczasową pomoc AI i wskazać rzeczywistą weryfikację studenta (oświadczenie studenta z 24.10.2026, że zweryfikował całość E1 i E2; szczegóły weryfikacji do dopisania w `report/sections/ai.tex`).
- [ ] Uzupełnić termin kalendarzowy E1 oraz wynik weryfikacji prowadzącego.

Potwierdzenie przekazania linku i akceptacja tematu: oświadczenie studenta z 24.10.2026 (bez zapisanej daty i formy przekazania). Wynik weryfikacji E1 przez prowadzącego: nie zapisano.

## E2 — środowisko, baseline i plan eksperymentów

Źródła: P00 §4, s. 3; P01 §3, s. 2–4.

- [x] Przygotować czytelną strukturę kodu i plik zależności/środowiska (`miner_rl/`, `requirements.txt`).
- [x] Zaktualizować instrukcję uruchomienia i sprawdzić czysty checkout bez ręcznych zmian ścieżek (24.10.2026, Claude Code: `git clone` do katalogu tymczasowego, `pytest` i `run_baselines.py` bez zmian ścieżek; użyto zainstalowanych już pakietów z `.venv`, bez świeżej instalacji `requirements.txt`).
- [x] Wprowadzić obsługę ziaren losowych, gdy wpływają na powtarzalność (`reset(seed)`, test powtarzalności).
- [x] Zaimplementować inicjalizację epizodu i krok środowiska (`miner_rl/env/mine_env.py`).
- [x] Zdefiniować obsługę poprawnych i niedozwolonych akcji.
- [x] Sprawdzić właściwe tematowi przypadki brzegowe, zakończenie i limit kroków (`tests/test_env.py`; `python -m pytest -q`: 188 passed, 21.10.2026, chmura Claude).
- [x] Opisać zdarzenia dynamiczne i potwierdzić zgodność kodu z raportem (raport, sekcja 4).
- [x] Uruchomić wielokrotne pełne epizody bez błędów (ewaluacja agentów bazowych, 1000 epizodów na mapę).
- [x] Zaimplementować adekwatny baseline i zapisać jego wyniki (`miner_rl/agents/baselines.py`, `results/`).
- [x] Przygotować plan porównań, wariantów i hiperparametrów (raport, sekcja „Plan eksperymentów”).
- [x] Wybrać metryki jakości i plan tabel/wykresów.
- [x] Zaplanować kilka uruchomień albo zapisać uzasadnienie odstępstwa (10 ziaren na wariant).
- [x] Przygotować pierwszą działającą wersję rozwiązania (`scripts/train.py`; pilotaż 24.10.2026: Q-learning i SARSA, 4 mapy, 3 ziarna, 50 000 epizodów, `results/train/pilot/`, raport sekcja „Eksperyment pilotażowy”).
- [x] Uzupełnić raport, dziennik decyzji i dokumentację AI w trakcie prac (raport i `report/sections/ai.tex` zaktualizowane 24.10.2026; weryfikacja studenta nadal do wpisania).
- [ ] Uzupełnić termin kalendarzowy E2 oraz wynik weryfikacji prowadzącego.

## E3 — finalne rozwiązanie i analiza

Źródła: P00 §4, s. 3; §8–10, s. 4–6; P01 §4, s. 4–5.

- [x] Zaimplementować i uruchomić właściwego agenta RL/DRL zgodnego z formalizacją (Q-learning i SARSA z maskowaniem akcji, zasady w wersji 3; 24.10.2026).
- [x] Zapisać hiperparametry, sposób reprezentacji Q-tab albo architekturę DRL (tablica Q jako słownik; konfiguracja końcowa w raporcie, sekcja „Stabilizacja polityki i kara za krok”, 25.10.2026; pakiet nagród przyjęty jako domyślny 24.10.2026).
- [x] Rozdzielić uczenie i ewaluację; zabezpieczyć niedozwolone akcje i nieskończone epizody (rozłączne ziarna treningu, walidacji i ewaluacji; maskowanie akcji; limit 300 kroków).
- [ ] Przeprowadzić eksperymenty zgodnie z planem i uczciwie porównać baseline (eksperyment 1 wykonany 24.10.2026, 10 ziaren, `results/train/glowny/`; stabilizacja i kara za krok 25.10.2026, `results/train/stabilizacja/`, `results/train/kara_krok/`; pozostały eksperymenty 2 i 3).
- [ ] Zachować dane lub skrypty wystarczające do odtworzenia wyników.
- [ ] Przygotować tabele/wykresy z osiami, jednostkami i legendami.
- [x] Pokazać przebieg uczenia, stabilność i niepowodzenia, nie tylko najlepszy wynik (krzywe uczenia, nieudane uruchomienia, analiza zachowania i map cieplnych w raporcie).
- [ ] Omówić wpływ zmian, ograniczenia i własne wnioski.
- [ ] Uzupełnić wszystkie wymagane części raportu oraz bibliografię.
- [ ] Potwierdzić zgodność raportu z aktualnym kodem i wynikami.
- [ ] Sprawdzić aktualne uruchamianie z czystego repo oraz kompilację LaTeX.
- [ ] Uzupełnić jawną sekcję „Wsparcie AI” w publicznym raporcie.
- [ ] Zachować pełne transkrypty AI z datami i nazwami modeli w dzienniku `ai-journal/` (część publicznego repozytorium).
- [ ] Dodać linki do całych rozmów poza narzędziami programistycznymi, jeśli narzędzia umożliwiają udostępnienie.
- [ ] Student samodzielnie sprawdził i umie wyjaśnić rozwiązanie, wyniki, ograniczenia oraz zakres pomocy AI; zapisać rzeczywisty zakres tej weryfikacji.
- [ ] Uzupełnić termin kalendarzowy E3 oraz wynik obrony/weryfikacji.

## E4 — poprawki i zamknięcie

Źródła: P00 §4, s. 3; P01 §5, s. 5–6.

- [ ] Zapisać uwagi prowadzącego i przypisać je do konkretnych poprawek.
- [ ] Poprawić wskazane błędy formalizacji i implementacji.
- [ ] Uzupełnić wymagane wyniki, wykresy i opisy.
- [ ] Zaktualizować raport oraz dokumentację AI.
- [ ] Sprawdzić uruchamianie i kompilację aktualnej wersji.
- [ ] Utrwalić poprawki rzeczywistymi commitami.
- [ ] Potwierdzić finalną zawartość repo i spełnienie uwag prowadzącego.
- [ ] Zapisać datę zamknięcia i wynik weryfikacji E4.

## Zasady aktualizowania checklisty

Odhaczać punkt dopiero po rzeczywistym wykonaniu i sprawdzeniu. Przy ważnych wynikach dopisać dowód: ścieżkę pliku, polecenie i wynik uruchomienia, commit albo potwierdzoną informację od prowadzącego. Działanie agenta nie stanowi automatycznie samodzielnej weryfikacji przez studenta. Przy błędzie ponownie otworzyć punkt i zapisać, co wymaga poprawy.

Nie zmieniać terminów, wymagań ani statusów dla stworzenia pozoru zgodności. Braki należy wykrywać i naprawiać przed oddaniem pracy. Lokalny charakter instrukcji agentów nie znosi obowiązków sprawozdawczych opisanych w P00.
