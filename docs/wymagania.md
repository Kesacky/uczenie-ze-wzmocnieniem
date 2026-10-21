# Wymagania projektu

Stan analizy: 21.10.2026. Dokument porządkuje wymagania zajęć „Uczenie się ze wzmocnieniem”, rok 2026/2027. Nie zastępuje zasad prowadzącego ani późniejszych ustaleń konsultacyjnych.

Źródła lokalne:

- **P00**: `P00_zasady_realizacji_projektu.pdf`, 6 stron.
- **P01**: `P01_checklista_projektu.pdf`, 7 stron; lista pomocnicza, podporządkowana zasadom P00 i ustaleniom z prowadzącym (P01 §1, s. 1).

Numery stron poniżej odpowiadają numerom wydrukowanym w dokumentach. Oryginalne PDF-y pozostają lokalnie: nie ustalono prawa do ich publicznego rozpowszechniania.

## Charakter pracy

Projekt ma obejmować samodzielne zaprojektowanie, implementację i analizę systemu RL dla wybranego problemu decyzyjnego lub sterowania. Samo uruchomienie gotowego algorytmu nie wystarcza. Student odpowiada za rozumienie i obronę całego rozwiązania, także elementów opracowanych przy pomocy AI. Praca jest domyślnie indywidualna; zespół wymaga wcześniejszego uzgodnienia z prowadzącym (P00 §1–2, s. 1; §9–11, s. 5–6).

Temat nie został jeszcze wybrany. Własny temat wymaga akceptacji prowadzącego najpóźniej w E1. Standardowa ścieżka to środowisko gridowe; ścieżki gry 2D, niestandardowego problemu i robota opisano osobno. Dobry projekt tablicowy może uzyskać maksimum punktów; DRL nie jest warunkiem maksymalnej oceny (P00 §6–7, s. 3–4).

## Macierz wymagań

| Obszar | Wymaganie / oczekiwany dowód | Źródło |
| --- | --- | --- |
| Repozytorium | Publiczne repo GitHub, czytelna nazwa, link przekazany prowadzącemu wskazaną przez niego drogą; bieżące aktualizacje i rzeczywista, systematyczna historia commitów. | P00 §2, s. 1–2; P01 §2.1, s. 1 |
| Zawartość repo | Kod, README z opisem i instrukcją uruchomienia, raport LaTeX, konfiguracja/skrypty odtworzenia środowiska, wyniki lub skrypty ich odtworzenia. | P00 §2, s. 1–2 |
| Publiczne materiały | Bez haseł, tokenów, kluczy, danych wrażliwych i materiałów bez prawa publikacji. Użycie niepublicznych danych/pliku uzgodnić z prowadzącym. | P00 §2, s. 1; P01 §2.1, s. 1 |
| Problem | Jasny cel agenta, sekwencyjne decyzje wpływające na przyszły stan, uzasadnienie zastosowania RL; problem nie może sprowadzać się do jednorazowej klasyfikacji/optymalizacji. | P00 §1, s. 1; P01 §2.2, s. 2 |
| Formalizacja | Agent, środowisko, stan/obserwacja, akcje, funkcja nagrody, warunki końca epizodu, ograniczenia, krok interakcji, miara sukcesu; wskazanie założeń do sprawdzenia po testach. | P01 §2.3, s. 2 |
| Uruchamianie | Czytelna struktura, aktualne README, osobny plik bibliotek/środowiska, uruchomienie z czystego checkoutu, brak nieopisanych ręcznych zmian ścieżek; ziarna losowe, jeśli istotne. | P01 §3.1, s. 3 |
| Środowisko | Inicjalizacja i krok zwracający stan, nagrodę, informację o zakończeniu; opis poprawnych i niepoprawnych akcji, zdarzeń dynamicznych; wielokrotne eksperymenty. | P01 §3.2, s. 3 |
| Poprawność środowiska | Sprawdzenie przypadków brzegowych właściwych tematowi, np. granice mapy, energia, pułapki, osiągnięcie celu; zgodność kodu z raportem. | P01 §3.2, s. 3 |
| Baseline | Adekwatny agent losowy/heurystyczny/inny, pełny epizod bez błędu, zapisane wyniki, uczciwa podstawa porównania. | P01 §3.3, s. 3 |
| Plan badań | Porównywane warianty, hiperparametry, metryki, baseline, kilka uruchomień albo uzasadnienie odstępstwa, plan tabel i wykresów. | P01 §3.4, s. 3–4 |
| Agent RL/DRL | Działająca implementacja zgodna z formalizacją; zapisane hiperparametry; oddzielne uczenie i ewaluacja; limit epizodu i obsługa niedozwolonych akcji. Q-tab: reprezentacja tablicy; DRL: architektura, wejścia, wyjścia i uczenie. | P01 §4.1, s. 4 |
| Eksperymenty | Realizacja planu i porównanie z baseline; odtwarzalne wyniki; przebieg uczenia, a nie wyłącznie najlepszy wynik; analiza niepowodzeń i wpływu zmian. | P01 §4.2, s. 4 |
| Prezentacja wyników | Tabele lub wykresy, opisane osie, jednostki i legendy, interpretacja oraz ograniczenia; wnioski wykraczające poza stwierdzenie, że agent działa. | P01 §4.2–4.3, s. 4–5; §7, s. 6 |
| Raport | LaTeX w repo, uzupełniany równolegle z pracą; zgodność z kodem, analiza i bibliografia, jawny opis pomocy AI. | P00 §2, s. 2; §8–9, s. 4–5; P01 §4.3, s. 5 |
| Obrona | Samodzielne wyjaśnienie problemu, formalizacji, środowiska, baseline, algorytmu, eksperymentów, wyników, ograniczeń i zakresu AI. | P00 §10, s. 6 |

## Raport LaTeX

Plik projektu: `report/main.tex`. Wymagane części (P00 §8, s. 4–5):

1. Opis problemu.
2. Uzasadnienie zastosowania RL.
3. Formalizacja RL/MDP.
4. Opis środowiska.
5. Opis agenta bazowego.
6. Opis zastosowanego algorytmu.
7. Plan eksperymentów.
8. Wyniki.
9. Analiza i wnioski.
10. Ograniczenia rozwiązania.
11. Instrukcja uruchomienia.
12. Wsparcie AI.
13. Bibliografia.

Nieuzupełnione fragmenty oznaczać jako niegotowe. Nie przedstawiać planu jako wykonanego badania, szkicu jako ukończonej implementacji ani testów agenta jako weryfikacji studenta.

## Dokumentowanie AI

Pomoc AI jest dopuszczalna pod warunkiem jawnego dokumentowania. P00 §9 (s. 5) wymaga:

- nazwy narzędzia lub modelu;
- daty użycia;
- celu i zakresu pomocy;
- informacji, które elementy student samodzielnie zweryfikował lub zmodyfikował;
- transkryptów rozmów z datą i nazwą modelu;
- dodatkowo linku do całej rozmowy poza narzędziami programistycznymi, o ile narzędzie umożliwia udostępnienie.

W tym projekcie pełny dziennik jest przechowywany lokalnie w `private/ai-journal/main.tex` wraz z plikami transkryptów i nie jest wysyłany do publicznego repo. Publiczny raport zawiera jawną sekcję „Wsparcie AI”. **Ignorowanie dziennika w Git nie zwalnia z obowiązku jego zachowania i udostępnienia prowadzącemu w uzgodniony sposób.** P00 nie określa, że pełne transkrypty muszą znajdować się w publicznym repo; sposób dostępu do lokalnego dziennika należy uzgodnić, jeśli prowadzący go nie wskazał.

Przechowywać dosłowne dostępne prompty i odpowiedzi; streszczenie może być dodatkiem, lecz nie zastępuje transkryptu. Brakujący fragment oznaczyć jako brak, a nie rekonstruować go jako rzekomo dosłowny zapis. Chronić dane wrażliwe, a konieczne redakcje wyraźnie oznaczać. Nie dokumentować ukrytych procesów rozumowania systemu jako wypowiedzi rozmowy.

Student musi rozumieć i umieć obronić także fragmenty opracowane z pomocą AI. Braki w dokumentacji mogą skutkować wyjaśnieniami, uzupełnieniem raportu, ponowną obroną fragmentu lub samodzielnym odtworzeniem części rozwiązania (P00 §9, s. 5; P01 §4.4, s. 5).

## Punktacja

| Element | Punkty |
| --- | ---: |
| Pomysł, kreatywność, trafność RL | 8 |
| Formalizacja: stan, akcje, nagrody, zakończenie | 10 |
| Środowisko i baseline | 8 |
| Algorytm RL/DRL i implementacja | 10 |
| Eksperymenty, porównania, wykresy, analiza | 8 |
| Raport LaTeX, repo, commity, obrona | 6 |
| **Razem** | **50** |

Źródło: P00 §7, s. 4. Dla niestandardowych, zespołowych i sprzętowych tematów prowadzący może doprecyzować kryteria. Szczególnie cenione są przemyślana nagroda, różne scenariusze, przewidywanie konsekwencji, uczciwy baseline, analiza niepowodzeń i powtarzalność (P01 §7, s. 6).

## Ryzyka obniżenia oceny

Brak formalizacji, baseline, porównań lub uruchamialności; wykresy bez opisu i analizy; przypadkowa nagroda; raport pisany dopiero na końcu i niezgodny z kodem; nieudokumentowane AI; brak systematycznych commitów (P01 §8, s. 7).

Każdy rozpoczęty tydzień opóźnienia wobec etapu oznacza obniżenie oceny o **10 punktów**; kary mogą dotyczyć osobno każdego opóźnionego etapu. Brak repo, dostępu, wymaganych commitów lub możliwości uruchomienia może być potraktowany jako niewykonanie etapu (P00 §5, s. 3). Harmonogram i checklisty: [etapy.md](etapy.md).
