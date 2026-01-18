from abc import ABC, abstractmethod
from typing import Tuple

class FaceRecognizer(ABC):
    @abstractmethod
    def load_known_faces(self, known_faces_dir: str) -> None:
        ...

    @abstractmethod
    def is_authorized(self, face_rgb, threshold: float) -> Tuple[bool, str, float]:
      
        ...