# Dziennik użycia AI

`log_ai.py` zapisuje przekazaną pełną wymianę prompt/odpowiedź, datę, narzędzie,
model, cel, zakres i stan samodzielnej weryfikacji studenta. Nie przechwytuje
automatycznie rozmów. Agent lub student musi dostarczyć pełne pliki wejściowe
po każdej wymianie; dłuższą rozmowę można archiwizować jako kolejne wpisy.
Nie zapisuj wewnętrznego toku rozumowania modelu.

Przykład w PowerShellu, z katalogu głównego projektu:

```powershell
python scripts/log_ai.py `
  --prompt-file private/prompt.txt `
  --response-file private/response.txt `
  --model "Rzeczywista nazwa modelu" `
  --date 2026-10-21 `
  --purpose "Omówienie formalizacji MDP" `
  --scope "Konsultacja stanu, akcji i nagrody"
```

Domyślne narzędzie to `Codex`; można podać `--tool`. Bez `--date` skrypt używa
`Europe/Warsaw`, jeśli systemowa baza stref czasowych jest dostępna. Gdy jej
brakuje, skrypt poprosi o jawną datę. Skrypt wymaga jedynie standardowej
biblioteki Pythona 3.10 lub nowszego.

`--verification` opisuje wyłącznie faktycznie wykonaną przez studenta weryfikację.
Bez tego parametru wpis wyraźnie oznacza ją jako niepotwierdzoną. Podanie
`--conversation-url` pozwala zachować link do pełnej rozmowy. Zgodnie z zasadami
kursu dodatkowy link dotyczy rozmowy poza narzędziami programistycznymi, o ile
użyte narzędzie umożliwia jej udostępnienie.

Dane trafiają do `private/ai-journal/entries/<data>-<id>/`:

- `prompt.txt` i `response.txt`: dokładne oryginalne bajty UTF-8, łącznie z białymi
  znakami i końcami linii;
- `metadata.json`: dane wpisu i sumy SHA-256 oryginałów;
- `entry.tex`: metadane oraz wczytanie obu tekstów przez `\VerbatimInput`.

Skrypt przebudowuje indeks `private/ai-journal/entries.tex`. Nie tworzy ani nie
modyfikuje `main.tex`. Główny dokument powinien używać kodowania UTF-8, polskich
fontów (dla XeLaTeX/LuaLaTeX przez `fontspec`, dla pdfLaTeX przez `fontenc`
i `inputenc`) oraz pakietu `fvextra` (zawijanie długich wierszy),
a następnie dołączać `\input{entries.tex}`. Kompilację wykonuj z katalogu
`private/ai-journal/`. Nie jest potrzebne `shell-escape`.

Surowe rozmowy przechowuj lokalnie zgodnie z `.gitignore`; przed przekazaniem
prowadzącemu przygotuj ich kopię. Skrypt **nie wykrywa ani nie usuwa sekretów**.
Nigdy nie podawaj modelowi haseł lub tokenów. Jeżeli transkrypt już zawiera sekret
albo dane wymagające ochrony, sporządź do udostępnienia kopię z jawnym oznaczeniem
redakcji i wyjaśnieniem jej zakresu; nie ukrywaj użycia AI ani jego istotnego
wpływu na pracę. Publiczną sekcję „Wsparcie AI” w raporcie uzupełniaj osobno.

Do testów można wskazać `--journal-dir` w katalogu tymczasowym. Równoległe zapisy
do jednego dziennika są blokowane na czas aktualizacji indeksu. Po przerwaniu
procesu sprawdź, czy nie działa inny zapisujący agent, zanim usuniesz ewentualną
osieroconą blokadę `.append.lock`.
