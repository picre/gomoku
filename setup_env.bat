@echo off
setlocal

echo ================================================
echo      Creating Conda environment
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
    echo GRESKA: Miniconda/Anaconda nije pronadjena.
    echo Instaliraj Minicondu ili promeni putanju u skripti.
    pause
    exit /b 1
)

echo Koristim:
echo %CONDA_BAT%
echo.

REM =================================================
REM Provera da li okruzenje postoji
REM =================================================
call "%CONDA_BAT%" env list | findstr /B "%ENV_NAME%" >nul

if errorlevel 1 (
    echo --- Kreiram okruzenje "%ENV_NAME%" ---
    call "%CONDA_BAT%" create -y -n %ENV_NAME% python=%PYTHON_VERSION%

    if errorlevel 1 (
        echo.
        echo GRESKA: Kreiranje okruzenja nije uspelo.
        pause
        exit /b 1
    )
) else (
    echo --- Okruzenje "%ENV_NAME%" vec postoji ---
)

echo.
echo --- Aktiviram okruzenje ---
call "%CONDA_BAT%" activate %ENV_NAME%

echo.
echo --- Upgrade pip ---
python -m pip install --upgrade pip

echo.
echo --- Instaliram requirements.txt ---
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo requirements.txt nije pronadjen.
)

echo.
echo --- Instaliram ipykernel ---
python -m pip install ipykernel

echo.
echo ================================================
echo              Setup zavrsen!
echo ================================================
echo.

echo Python:
python --version

echo.
echo Aktivno okruzenje:
echo %CONDA_DEFAULT_ENV%
echo.

cmd /k call "%CONDA_BAT%" activate %ENV_NAME%