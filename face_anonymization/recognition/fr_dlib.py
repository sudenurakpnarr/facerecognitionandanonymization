from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import face_recognition

from .base import FaceRecognizer


@dataclass
class KnownFace:
    name: str
    encoding: np.ndarray


class DlibFaceRecognizer(FaceRecognizer):
    def __init__(self):
        self.known: List[KnownFace] = []

    def load_known_faces(self, known_faces_dir: str) -> None:
        """
        Loads known face images from the given directory and computes face encodings.
        The directory path is resolved relative to the project root.
        """

        # fr_dlib.py -> recognition -> face_anonymization -> cv_proje (project root)
        project_root = Path(__file__).resolve().parents[3]
        kdir = (project_root / known_faces_dir).resolve()

        if not kdir.exists():
            raise FileNotFoundError(f"KNOWN_FACES_DIR not found: {kdir}")

        exts = {".jpg", ".jpeg", ".png"}
        files = [p for p in kdir.iterdir() if p.suffix.lower() in exts]

        if not files:
            raise FileNotFoundError(f"No images found in known_faces directory: {kdir}")

        self.known.clear()

        for img_path in files:
            # Use filename (without extension) as identity label
            name = img_path.stem
            img = face_recognition.load_image_file(str(img_path))
            encodings = face_recognition.face_encodings(img)

            # Skip images where no face is detected
            if not encodings:
                print(f"[WARNING] No face detected in {img_path.name}, skipping...")
                continue

            self.known.append(KnownFace(name=name, encoding=encodings[0]))
            print(f"[LOAD] Loaded face: {name} from {img_path.name}")

        if not self.known:
            raise RuntimeError(
                "No valid face encodings could be extracted from known_faces images."
            )
        print(f"[LOAD] Total {len(self.known)} known faces loaded successfully.")

    def is_authorized(self, face_rgb, threshold: float) -> Tuple[bool, str, float]:
        """
        Determines whether the given face belongs to an authorized identity.

        Returns:
            authorized (bool): True if the face matches a known identity
            name (str): matched identity name or 'unknown'
            score (float): similarity-based confidence score
        """

        encodings = face_recognition.face_encodings(face_rgb)
        if not encodings:
            return False, "unknown", 0.0

        encoding = encodings[0]

        known_encodings = [k.encoding for k in self.known]
        distances = face_recognition.face_distance(known_encodings, encoding)

        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        best_name = self.known[best_index].name

        # Smaller distance indicates a better match
        authorized = best_distance <= float(threshold)

        # Convert distance to a simple similarity score (0–1 range)
        score = max(0.0, 1.0 - best_distance)

        if authorized:
            return True, best_name, score

        return False, "unknown", score