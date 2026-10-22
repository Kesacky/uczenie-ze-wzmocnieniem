@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto dependencies
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m venv .venv
    goto checkvenv
)
where python >nul 2>nul
if errorlevel 1 (
    echo Nie znaleziono Pythona. Zainstaluj Python 3 i uruchom gre ponownie.
    pause
    exit /b 1
)
python -m venv .venv
:checkvenv
if not exist ".venv\Scripts\python.exe" (
    echo Nie udalo sie utworzyc srodowiska Python.
    pause
    exit /b 1
)
:dependencies
if not exist ".venv\requirements.installed" goto install
fc /b requirements.txt .venv\requirements.installed >nul 2>nul
if not errorlevel 1 goto play
:install
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo Nie udalo sie zainstalowac zaleznosci.
    pause
    exit /b 1
)
copy /y requirements.txt .venv\requirements.installed >nul
if errorlevel 1 (
    echo Nie udalo sie zapisac kopii requirements.txt.
    pause
    exit /b 1
)
:play
.venv\Scripts\python.exe scripts\play.py %*
set "game_result=%errorlevel%"
if not "%game_result%"=="0" (
    echo Gra zakonczyla sie bledem. Kod: %game_result%
    pause
)
exit /b %game_result%
