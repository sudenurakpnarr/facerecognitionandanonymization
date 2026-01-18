from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .base import FaceRecognizer
from face_anonymization.detection.mtcnn_detector import MTCNNDetector


@dataclass
class KnownFace:
    name: str
    embedding: np.ndarray


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = vec.astype(np.float32)
    norm = float(np.linalg.norm(vec) + eps)
    return vec / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings are already normalized, but keep safe
    a = _l2_normalize(a)
    b = _l2_normalize(b)
    return float(np.dot(a, b))


def _preprocess_arcface(face_bgr: np.ndarray) -> np.ndarray:
    face = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)

    # ArcFace normalization
    face_rgb = (face_rgb - 127.5) / 127.5

    blob = np.transpose(face_rgb, (2, 0, 1))[None, ...]
    return blob.astype(np.float32)


class ArcFaceONNXRecognizer(FaceRecognizer):
    def __init__(self, model_path: str = "models/arcface.onnx", min_confidence: float = 0.90):
        self.model_path = model_path
        self.known: List[KnownFace] = []

        # used only for building the known-face gallery (cropping from static images)
        self.detector = MTCNNDetector(min_confidence=min_confidence)

        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None

    def _get_session(self) -> ort.InferenceSession:
        if self._session is not None:
            return self._session

        project_root = Path(__file__).resolve().parents[2]
        mpath = (project_root / self.model_path).resolve()

        if not mpath.exists():
            raise FileNotFoundError(
                f"ArcFace ONNX model not found: {mpath}\n"
                f"(Expected at: {self.model_path})"
            )

        sess = ort.InferenceSession(str(mpath), providers=["CPUExecutionProvider"])
        self._session = sess
        self._input_name = sess.get_inputs()[0].name
        return self._session

    def _embed_face(self, face_bgr: np.ndarray) -> np.ndarray:
        sess = self._get_session()
        assert self._input_name is not None

        # reject tiny crops (these cause unstable embeddings)
        h, w = face_bgr.shape[:2]
        if h < 40 or w < 40:
            raise ValueError("Face crop too small for embedding.")

        blob = _preprocess_arcface(face_bgr)
        out = sess.run(None, {self._input_name: blob})[0]  # (1, D)
        emb = out[0].astype(np.float32)
        return _l2_normalize(emb)

    def _crop_first_face(self, img_bgr: np.ndarray, pad: float = 0.20) -> np.ndarray | None:
        """
        Crops the first detected face from a static image (known_faces gallery).
        Adds padding around the box to make cropping more stable.
        """
        boxes = self.detector.detect(img_bgr)
        if not boxes:
            return None

        x, y, w, h = boxes[0]
        if w <= 0 or h <= 0:
            return None

        h_img, w_img = img_bgr.shape[:2]

        # padding
        px = int(w * pad)
        py = int(h * pad)

        x1 = max(x - px, 0)
        y1 = max(y - py, 0)
        x2 = min(x + w + px, w_img)
        y2 = min(y + h + py, h_img)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = img_bgr[y1:y2, x1:x2]
        # reject tiny crops
        if crop.shape[0] < 40 or crop.shape[1] < 40:
            return None
        return crop

    def load_known_faces(self, known_faces_dir: str) -> None:
        project_root = Path(__file__).resolve().parents[2]
        kdir = (project_root / known_faces_dir).resolve()

        if not kdir.exists():
            raise FileNotFoundError(f"KNOWN_FACES_DIR not found: {kdir}")

        exts = {".jpg", ".jpeg", ".png"}
        files = [p for p in kdir.iterdir() if p.suffix.lower() in exts]
        if not files:
            raise FileNotFoundError(f"No images found in known_faces directory: {kdir}")

        self.known.clear()

        for img_path in files:
            name = img_path.stem
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            face_crop = self._crop_first_face(img_bgr, pad=0.20)
            if face_crop is None:
                continue

            try:
                emb = self._embed_face(face_crop)
            except ValueError:
                continue

            self.known.append(KnownFace(name=name, embedding=emb))
            print(f"[LOAD] Loaded face: {name} from {img_path.name}")

        if not self.known:
            raise RuntimeError("No valid embeddings extracted from known_faces images.")
        print(f"[LOAD] Total {len(self.known)} known faces loaded successfully.")

    def is_authorized(self, face_rgb, threshold: float) -> Tuple[bool, str, float]:
        if not self.known:
            return False, "unknown", 0.0

        # face_rgb comes from webcam crop already; convert to BGR for preprocessing
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

        try:
            emb = self._embed_face(face_bgr)
        except ValueError:
            # too small / bad crop -> treat as unknown
            return False, "unknown", 0.0

        best_name = "unknown"
        best_sim = -1.0

        # Tüm similarity değerlerini göster
        print(f"  [SIMILARITY CHECK] Comparing with {len(self.known)} known faces:")
        for k in self.known:
            sim = _cosine_similarity(emb, k.embedding)
            print(f"    - {k.name}: similarity={sim:.4f}")
            if sim > best_sim:
                best_sim = sim
                best_name = k.name

        print(f"  [BEST MATCH] name='{best_name}', similarity={best_sim:.4f}, threshold={threshold:.4f}")
        
        authorized = best_sim >= float(threshold)
        # Her zaman best_name'i döndür (threshold'u geçmese bile)
        # Böylece known_faces'teki kişileri tanıyabiliriz
        return authorized, best_name, float(best_sim)