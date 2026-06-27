@echo off
echo ================================================
echo      Creating Python virtual environment
echo ================================================

REM --- Putanja do Python-a 3.12 (promeni ako treba) ---
set PYTHON=python

echo.
echo --- Kreiram .venv ---
%PYTHON% -m venv .venv
IF %ERRORLEVEL% NEQ 0 (
    echo GRESKA: Ne mogu da kreiram virtual environment.
    pause
    exit /b
)

echo.
echo --- Aktiviram .venv ---
call .venv\Scripts\activate

echo.
echo --- Upgrade pip ---
python -m pip install --upgrade pip

echo.
echo --- Instaliram pakete iz requirements.txt ---
pip install -r requirements.txt

echo.
echo --- Instaliram ipykernel za VS Code ---
python -m pip install ipykernel

echo.
echo ================================================
echo      Setup zavrsen!
echo ================================================
echo.
echo Venv je aktivan. Mozes sada da kucas komande ovde.
echo.

cmd /k
