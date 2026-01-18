import cv2
import os
from pathlib import Path

# Known faces klasörünü bul
project_root = Path(__file__).resolve().parent
known_faces_dir = project_root / "known_faces"
known_faces_dir.mkdir(exist_ok=True)

print(f"[INFO] Fotoğraf {known_faces_dir} klasörüne kaydedilecek")
print("[INFO] Webcam açılıyor...")
print("[INFO] 'SPACE' tuşuna basarak fotoğraf çekin, 'q' tuşu ile çıkın")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[HATA] Webcam açılamadı!")
    exit(1)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Frame'i göster
    cv2.putText(frame, "SPACE: Fotoğraf Cek | Q: Cikis", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Fotoğraf Çek - SPACE tuşuna basın", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # SPACE tuşu
        # Fotoğrafı kaydet
        filename = known_faces_dir / "kullanici.jpeg"
        cv2.imwrite(str(filename), frame)
        print(f"[OK] Fotoğraf kaydedildi: {filename}")
        print(f"[OK] Toplam {len(list(known_faces_dir.glob('*.jpeg')) + list(known_faces_dir.glob('*.jpg')) + list(known_faces_dir.glob('*.png')))} fotoğraf known_faces klasöründe")
        break
    elif key == ord('q'):
        print("[INFO] Çıkılıyor...")
        break

cap.release()
cv2.destroyAllWindows()
print("[OK] Tamamlandı!")
