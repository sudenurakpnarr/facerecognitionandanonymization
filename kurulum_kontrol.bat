@echo off
echo Python versiyonlarini kontrol ediyorum...
echo.

where python3.11 >nul 2>&1
if %errorlevel% == 0 (
    echo [BULUNDU] Python 3.11
    python3.11 --version
) else (
    echo [BULUNAMADI] Python 3.11
)

where python3.12 >nul 2>&1
if %errorlevel% == 0 (
    echo [BULUNDU] Python 3.12
    python3.12 --version
) else (
    echo [BULUNAMADI] Python 3.12
)

where python >nul 2>&1
if %errorlevel% == 0 (
    echo.
    echo [MEVCUT] Python:
    python --version
)

pause
