# Etapy i lista kontrolna

Stan na 21.10.2026: przeczytano oba PDF-y, spisano wymagania, utworzono publiczne repozytorium i szablon raportu. Kompilacja raportu we wbudowanym edytorze zakończyła się powodzeniem. Temat nie został wybrany, formalizacja nie jest gotowa, nie potwierdzono przekazania linku prowadzącemu. **E1 nie jest ukończony.**

Publiczne repozytorium: [Kesacky/uczenie-ze-wzmocnieniem](https://github.com/Kesacky/uczenie-ze-wzmocnieniem). Samo umieszczenie linku w dokumentacji nie oznacza przekazania go prowadzącemu ani zaliczenia etapu.

Źródła: P00 = lokalny `P00_zasady_realizacji_projektu.pdf`; P01 = lokalny `P01_checklista_projektu.pdf`. Pełne wymagania i punktacja: [wymagania.md](wymagania.md).

## Harmonogram

| Etap | Termin względny | Data kalendarzowa | Wymagany rezultat | Status |
| --- | --- | --- | --- | --- |
| E1 | do końca P2 | do potwierdzenia | temat, link do repo, opis problemu, wstępna formalizacja | nieukończony |
| E2 | do P4 | do potwierdzenia | środowisko, baseline, plan badań, pierwsza działająca wersja | nierozpoczęty |
| E3 | do P6 | do potwierdzenia | finalny projekt, raport, eksperymenty, wyniki i analiza | nierozpoczęty |
| E4 | P7 | do potwierdzenia | poprawki wskazane przez prowadzącego, domknięcie | oczekuje na uwagi |

Źródło: P00 §3–4, s. 2–3. PDF-y nie podają dat kalendarzowych; uzupełnić je na podstawie planu zajęć lub informacji prowadzącego. E4 służy poprawkom, a nie wykonywaniu projektu od początku (P01 §5, s. 5–6).

Każdy rozpoczęty tydzień opóźnienia może kosztować 10 punktów, osobno dla każdego etapu (P00 §5, s. 3). Przed etapem należy sprawdzić repo, raport, uruchamialność i gotowość do wyjaśnienia pracy.

## E1 — temat, repozytorium, formalizacja

Źródła: P00 §2, s. 1–2; §4, s. 3; P01 §2, s. 1–2.

- [x] Przeczytać zasady i checklistę projektu.
- [x] Spisać wymagania i etapy wraz ze źródłami.
- [x] Potwierdzić, że publiczne repo GitHub zostało utworzone i jest dostępne (21.10.2026, GitHub potwierdził widoczność `public`).
- [ ] Przekazać link prowadzącemu wskazaną przez niego drogą i zapisać datę/czynność.
- [x] Dodać README z prawdziwym opisem aktualnego stanu (`README.md`).
- [x] Dodać szkielet sprawozdania `report/main.tex` i sprawdzić jego kompilację (21.10.2026, kompilator edytora: sukces).
- [x] Wykonać pierwszy commit rzeczywiście utworzonych materiałów (commit inicjujący dokumentację i narzędzie dziennika, 21.10.2026).
- [x] Sprawdzić przed publikacją, czy brak sekretów, danych wrażliwych i materiałów bez prawa rozpowszechniania; oryginalne PDF-y zostawić lokalnie (kontrola plików i `git check-ignore`).
- [ ] Wybrać i jednoznacznie opisać temat; uzgodnić własny temat z prowadzącym.
- [ ] Zapisać cel agenta i uzasadnienie sekwencyjnego charakteru problemu oraz zastosowania RL.
- [ ] Opisać agenta, środowisko i stan/obserwację.
- [ ] Zdefiniować dostępne akcje, nagrody i warunki końca epizodu.
- [ ] Opisać ograniczenia, pojedynczy krok interakcji i mierzalną definicję sukcesu.
- [ ] Wskazać niepewne założenia formalizacji do weryfikacji po testach.
- [ ] Udokumentować dotychczasową pomoc AI i wskazać rzeczywistą weryfikację studenta.
- [ ] Uzupełnić termin kalendarzowy E1 oraz wynik weryfikacji prowadzącego.

Potwierdzenie przekazania linku: **brak**. Temat/akceptacja: **brak**. Wynik weryfikacji E1: **brak**.

## E2 — środowisko, baseline i plan eksperymentów

Źródła: P00 §4, s. 3; P01 §3, s. 2–4.

- [ ] Przygotować czytelną strukturę kodu i plik zależności/środowiska.
- [ ] Zaktualizować instrukcję uruchomienia i sprawdzić czysty checkout bez ręcznych zmian ścieżek.
- [ ] Wprowadzić obsługę ziaren losowych, gdy wpływają na powtarzalność.
- [ ] Zaimplementować inicjalizację epizodu i krok środowiska.
- [ ] Zdefiniować obsługę poprawnych i niedozwolonych akcji.
- [ ] Sprawdzić właściwe tematowi przypadki brzegowe, zakończenie i limit kroków.
- [ ] Opisać zdarzenia dynamiczne i potwierdzić zgodność kodu z raportem.
- [ ] Uruchomić wielokrotne pełne epizody bez błędów.
- [ ] Zaimplementować adekwatny baseline i zapisać jego wyniki.
- [ ] Przygotować plan porównań, wariantów i hiperparametrów.
- [ ] Wybrać metryki jakości i plan tabel/wykresów.
- [ ] Zaplanować kilka uruchomień albo zapisać uzasadnienie odstępstwa.
- [ ] Przygotować pierwszą działającą wersję rozwiązania.
- [ ] Uzupełnić raport, dziennik decyzji i dokumentację AI w trakcie prac.
- [ ] Uzupełnić termin kalendarzowy E2 oraz wynik weryfikacji prowadzącego.

## E3 — finalne rozwiązanie i analiza

Źródła: P00 §4, s. 3; §8–10, s. 4–6; P01 §4, s. 4–5.

- [ ] Zaimplementować i uruchomić właściwego agenta RL/DRL zgodnego z formalizacją.
- [ ] Zapisać hiperparametry, sposób reprezentacji Q-tab albo architekturę DRL.
- [ ] Rozdzielić uczenie i ewaluację; zabezpieczyć niedozwolone akcje i nieskończone epizody.
- [ ] Przeprowadzić eksperymenty zgodnie z planem i uczciwie porównać baseline.
- [ ] Zachować dane lub skrypty wystarczające do odtworzenia wyników.
- [ ] Przygotować tabele/wykresy z osiami, jednostkami i legendami.
- [ ] Pokazać przebieg uczenia, stabilność i niepowodzenia, nie tylko najlepszy wynik.
- [ ] Omówić wpływ zmian, ograniczenia i własne wnioski.
- [ ] Uzupełnić wszystkie wymagane części raportu oraz bibliografię.
- [ ] Potwierdzić zgodność raportu z aktualnym kodem i wynikami.
- [ ] Sprawdzić aktualne uruchamianie z czystego repo oraz kompilację LaTeX.
- [ ] Uzupełnić jawną sekcję „Wsparcie AI” w publicznym raporcie.
- [ ] Zachować pełne transkrypty AI z datami i nazwami modeli w lokalnym `private/ai-journal`; udostępnić prowadzącemu w uzgodniony sposób.
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

Nie zmieniać terminów, wymagań ani statusów dla stworzenia pozoru zgodności. Braki należy wykrywać i naprawiać przed oddaniem pracy. Lokalny charakter instrukcji agentów lub dziennika AI nie znosi obowiązków sprawozdawczych opisanych w P00.
