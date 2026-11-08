@echo off
setlocal
REM UTF-8 kodna strana da bi se dijakritici ispravno prikazali u konzoli
chcp 65001 >nul

echo ================================================
echo      Pravljenje Conda okruženja
echo ================================================

REM =================================================
REM Konfiguracija
REM =================================================
set ENV_NAME=gomoku
set PYTHON_VERSION=3.12

echo.

REM =================================================
REM Pronalazi conda.bat
REM =================================================
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    set CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat
) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    set CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat
) else (
    echo GREŠKA: Miniconda/Anaconda nije pronađena.
    echo Potrebno je instalirati Minicondu ili promeniti putanju u skripti.
    pause
    exit /b 1
)

echo Koristi se:
echo %CONDA_BAT%
echo.

REM =================================================
REM Provera da li okruženje postoji
REM =================================================
call "%CONDA_BAT%" env list | findstr /B "%ENV_NAME%" >nul

if errorlevel 1 (
    echo --- Pravi se okruženje "%ENV_NAME%" ---
    call "%CONDA_BAT%" create -y -n %ENV_NAME% python=%PYTHON_VERSION%

    if errorlevel 1 (
        echo.
        echo GREŠKA: Pravljenje okruženja nije uspelo.
        pause
        exit /b 1
    )
) else (
    echo --- Okruženje "%ENV_NAME%" već postoji ---
)

echo.
echo --- Aktivira se okruženje ---
call "%CONDA_BAT%" activate %ENV_NAME%

echo.
echo --- Ažurira se pip ---
python -m pip install --upgrade pip

echo.
echo --- Instaliraju se zavisnosti iz requirements.txt ---
if exist requirements.txt (
    python -m pip install -r requirements.txt
) else (
    echo requirements.txt nije pronađen.
)

echo.
echo ================================================
echo              Podešavanje završeno!
echo ================================================
echo.

echo Python:
python --version

echo.
echo Aktivno okruženje:
echo %CONDA_DEFAULT_ENV%
echo.

cmd /k call "%CONDA_BAT%" activate %ENV_NAME%
