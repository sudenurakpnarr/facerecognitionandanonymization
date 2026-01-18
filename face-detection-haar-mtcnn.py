import cv2
from mtcnn import MTCNN

detector = MTCNN()

def blur_face(frame, x, y, w, h):
    H, W = frame.shape[:2]
    x = max(0, x); y = max(0, y)
    x2 = min(W, x + max(0, w))
    y2 = min(H, y + max(0, h))

    face = frame[y:y2, x:x2]
    if face.size == 0:
        return frame

    blurred = cv2.GaussianBlur(face, (99, 99), 30)
    frame[y:y2, x:x2] = blurred
    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)

    for det in detections:
        x, y, w, h = det["box"]
        frame = blur_face(frame, x, y, w, h)

    cv2.imshow("MTCNN Face Detection + Blur", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
