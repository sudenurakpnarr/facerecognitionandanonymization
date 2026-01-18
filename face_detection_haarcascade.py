import cv2
from mtcnn import MTCNN
import face_recognition
import os

detector = MTCNN()

KNOWN_FACES_DIR = "known_faces"
known_encodings = []
known_names = []

for file in os.listdir(KNOWN_FACES_DIR):
    if file.endswith(".jpg") or file.endswith(".png"):
        image = face_recognition.load_image_file(
            os.path.join(KNOWN_FACES_DIR, file)
        )
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(file.split(".")[0])


def blur_face(frame, x, y, w, h):
    face = frame[y:y+h, x:x+w]
    if face.size == 0:
        return frame

    blurred = cv2.GaussianBlur(face, (99, 99), 30)
    frame[y:y+h, x:x+w] = blurred
    return frame


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for det in detections:
        x, y, w, h = det["box"]
        name = "Unknown"

        for (top, right, bottom, left), encoding in zip(
            face_locations, face_encodings
        ):
            if abs(left - x) < 50 and abs(top - y) < 50:
                matches = face_recognition.compare_faces(
                    known_encodings, encoding, tolerance=0.5
                )
                if True in matches:
                    name = known_names[matches.index(True)]

        if name == "Unknown":
            frame = blur_face(frame, x, y, w, h)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, name, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

    cv2.imshow("MTCNN Face Detection + Identification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()