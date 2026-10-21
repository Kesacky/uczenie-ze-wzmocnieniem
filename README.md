# Uczenie się ze wzmocnieniem - projekt 2026/2027

Repozytorium projektu z przedmiotu „Uczenie się ze wzmocnieniem”, WEiI PRz.

**Stan na 21.10.2026:** przygotowano strukturę dokumentacji, zestawienie wymagań,
szablon raportu LaTeX oraz narzędzie do zapisywania rozmów z AI. Temat projektu
nie został jeszcze wybrany; nie ma implementacji środowiska, agentów ani wyników.
Etap E1 pozostaje otwarty.

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

Źródłem wymagań są „Zasady realizacji projektu” i „Lista kontrolna projektu”,
2026/2027. Oryginały materiałów kursowych są przechowywane lokalnie; nie są
publikowane bez potwierdzenia prawa do ich udostępniania.

## Uruchomienie dokumentacji

Repozytorium nie ma jeszcze programu RL do uruchomienia. Instrukcja programu,
plik zależności oraz konfiguracje eksperymentów zostaną dodane wraz z implementacją.

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

Pełny dziennik LaTeX i oryginalne teksty wymian są przechowywane lokalnie
w `private/ai-journal/`, poza publicznym Gitem. Należy zachować prywatną kopię
zapasową i udostępnić transkrypty prowadzącemu w uzgodniony sposób. PDF-y nie
precyzują, że pełne transkrypty muszą być opublikowane w publicznym repozytorium.
Rozmowy spoza narzędzi programistycznych należy dodatkowo udostępnić przez link,
o ile narzędzie daje taką możliwość. Sposób przekazania transkryptów prowadzącemu
nie został jeszcze potwierdzony.

Skrypt zapisuje dostarczone teksty; nie przechwytuje automatycznie rozmów.
Każda nowa rozmowa wymaga zapisania jej pełnego przebiegu i aktualizacji raportu.

## Najbliższy krok

Wybrać temat, opisać cel agenta i wstępne MDP, potwierdzić wymagane uzgodnienia
z prowadzącym oraz przekazać mu link do repozytorium. Daty kalendarzowe bloków
zajęć należy wpisać po ich potwierdzeniu.
