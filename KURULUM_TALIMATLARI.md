# Python Kurulum Talimatları

## Adım 1: Python 3.11 veya 3.12 Kurulumu

### Yöntem 1: Python.org'dan İndirme (ÖNERİLEN)

1. **İndirme:**
   - Tarayıcınızda şu adrese gidin: https://www.python.org/downloads/
   - "Python 3.11.x" veya "Python 3.12.x" için "Download" butonuna tıklayın
   - **Windows installer (64-bit)** seçeneğini indirin

2. **Kurulum:**
   - İndirilen `.exe` dosyasını çalıştırın
   - **ÖNEMLİ:** "Add Python to PATH" kutusunu **mutlaka işaretleyin** ✓
   - "Install Now" butonuna tıklayın
   - Kurulum tamamlanana kadar bekleyin

3. **Kontrol:**
   - Yeni bir PowerShell/CMD penceresi açın
   - Şu komutu çalıştırın:
   ```bash
   python --version
   ```
   - Python 3.11.x veya 3.12.x görünmeli

### Yöntem 2: Microsoft Store

1. Microsoft Store'u açın
2. "Python 3.11" veya "Python 3.12" arayın
3. Microsoft'tan Python'u kurun
4. Kurulumdan sonra yeni bir terminal açın

---

## Adım 2: Gerekli Paketleri Kurma

Kurulum tamamlandıktan sonra, proje dizininde şu komutları çalıştırın:

```bash
# pip'i güncelle
python -m pip install --upgrade pip

# Gerekli paketleri kur
python -m pip install opencv-python mtcnn numpy tensorflow onnxruntime
```

**VEYA** hazırladığım batch dosyasını çalıştırın:
```bash
paketleri_kur.bat
```

---

## Adım 3: Uygulamayı Çalıştırma

Proje dizininde:

```bash
cd cv_proje
python -m face_anonymization.pipeline.webcam
```

**VEYA** doğrudan:

```bash
python cv_proje\face_anonymization\pipeline\webcam.py
```

---

## Sorun Giderme

### Python bulunamıyor hatası:
- Kurulum sırasında "Add Python to PATH" seçeneğini işaretlediğinizden emin olun
- Bilgisayarı yeniden başlatın
- Yeni bir terminal penceresi açın

### Paket kurulum hataları:
- Python 3.11 veya 3.12 kullandığınızdan emin olun (3.14 desteklenmiyor)
- pip'i güncelleyin: `python -m pip install --upgrade pip`
- İnternet bağlantınızı kontrol edin

### Kamera açılmıyor:
- `config.py` dosyasında `CAMERA_INDEX = 0` değerini kontrol edin
- Farklı bir kamera varsa `CAMERA_INDEX = 1` deneyin

---

## Notlar

- **Python 3.14 kullanmayın!** TensorFlow ve onnxruntime henüz desteklemiyor
- Kamera erişim izinlerini kontrol edin
- `known_faces` klasöründe Tarkan'ın fotoğrafları olmalı
- Uygulamayı kapatmak için 'q' tuşuna basın
