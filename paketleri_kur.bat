@echo off
echo Gerekli paketleri kuruyorum...
echo.

REM Python versiyonunu kontrol et
python --version
if %errorlevel% neq 0 (
    echo HATA: Python bulunamadi!
    pause
    exit /b 1
)

echo.
echo pip'i guncelliyorum...
python -m pip install --upgrade pip

echo.
echo Paketleri kuruyorum...
python -m pip install opencv-python mtcnn numpy tensorflow onnxruntime

echo.
echo Kurulum tamamlandi!
pause
