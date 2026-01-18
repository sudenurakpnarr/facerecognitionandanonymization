import time
import cv2

from face_anonymization.config import (
    CAMERA_INDEX,
    KNOWN_FACES_DIR,
    AUTH_THRESHOLD,
    BLUR_KERNEL_SIZE,
    DRAW_BOXES,
    SHOW_FPS,
)

from face_anonymization.detection.mtcnn_detector import MTCNNDetector
from face_anonymization.anonymization.blur import apply_blur
from face_anonymization.recognition.arcface_onnx import ArcFaceONNXRecognizer


def run_webcam():
    detector = MTCNNDetector(min_confidence=0.70)
    recognizer = ArcFaceONNXRecognizer(model_path="models/arcface.onnx", min_confidence=0.90)

    recognizer.load_known_faces(KNOWN_FACES_DIR)
    print(f"[OK] Loaded authorized faces from: {KNOWN_FACES_DIR}")
    print(f"[INFO] Total known faces loaded: {len(recognizer.known)}")
    for kf in recognizer.known:
        print(f"  - {kf.name}")
    print("[INFO] Press 'q' to quit.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    prev = time.time()
    fps = 0.0
    frame_id = 0
    blur_applied_this_frame = False  # Blur durumunu takip et

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        now = time.time()
        dt = now - prev
        prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        boxes = detector.detect(frame)

        for (x, y, w, h) in boxes:
            face_bgr = frame[y:y + h, x:x + w]
            if face_bgr.size == 0:
                continue

            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

            authorized, name, score = recognizer.is_authorized(face_rgb, AUTH_THRESHOLD)

            status = "AUTHORIZED" if authorized else "ANONYMIZED"
            print(
                f"[FRAME {frame_id}] name={name} | score={score:.3f} | threshold={AUTH_THRESHOLD:.3f} | authorized={authorized} | status={status} | "
                f"box=({x},{y},{w},{h})"
            )

            # BLUR KONTROLÜ: 
            # - Eğer name "tarkan" ile başlıyorsa → blur YAPMA (Tarkan known_faces'te var, score ne olursa olsun)
            # - Eğer authorized=True ise → blur YAPMA (known_faces'te tanımlı, yeterli score)
            # - Eğer authorized=False VEYA name="unknown" ise → blur yap (known_faces'te yok veya düşük score)
            # Her yüz için koordinatlarına göre ayrı ayrı kontrol edilir
            
            # Basit mantık: known_faces'te tanımlı ise blur yapma, değilse blur yap
            # - authorized=True → blur yapma (known_faces'te tanımlı, yeterli benzerlik)
            # - authorized=False → blur yap (known_faces'te yok veya düşük benzerlik)
            
            if authorized:
                # Known_faces'te tanımlı → blur yapma
                should_blur = False
            else:
                # Known_faces'te yok → blur yap
                should_blur = True
            
            # Blur uygula
            if should_blur:
                print(f"  [BEFORE BLUR] should_blur={should_blur}, applying blur to coordinates: x={x}, y={y}, w={w}, h={h}")
                frame = apply_blur(frame, (x, y, w, h), kernel_size=BLUR_KERNEL_SIZE)
                blur_applied_this_frame = True
                print(f"  [AFTER BLUR] BLUR UYGULANDI (koordinat: x={x}, y={y}, w={w}, h={h})")
            else:
                print(f"  [NO BLUR] BLUR YAPILMADI (koordinat: x={x}, y={y}, w={w}, h={h})")

            if DRAW_BOXES:
                label = f"{name} ({score:.2f})" if authorized else f"unknown ({score:.2f})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(
                    frame,
                    label,
                    (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        if SHOW_FPS:
            blur_status = "BLUR VAR" if blur_applied_this_frame else "BLUR YOK"
            blur_color = (0, 0, 255) if blur_applied_this_frame else (0, 255, 0)
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | {blur_status}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                blur_color,
                2,
            )
        
        # Blur durumunu sıfırla (bir sonraki frame için)
        blur_applied_this_frame = False

        cv2.imshow("Identity-Aware Face Anonymization", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()