from typing import List, Tuple, Dict

import cv2
from mtcnn import MTCNN

Box = Tuple[int, int, int, int]  # (x, y, w, h)


class MTCNNDetector:
    def __init__(self, min_confidence: float = 0.90):
        self.detector = MTCNN()
        self.min_confidence = float(min_confidence)

    def detect(self, frame_bgr) -> List[Box]:
        """
        Input:  OpenCV BGR frame
        Output: list of face boxes (x, y, w, h)
        """
        if frame_bgr is None:
            return []

        # Very small frames can break MTCNN
        h, w = frame_bgr.shape[:2]
        if h < 60 or w < 60:
            return []

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            detections: List[Dict] = self.detector.detect_faces(frame_rgb)
        except ValueError:
            # MTCNN sometimes produces empty internal batches (known issue)
            return []

        boxes: List[Box] = []
        for det in detections:
            conf = det.get("confidence", 0.0)
            if conf < self.min_confidence:
                continue

            x, y, bw, bh = det.get("box", (0, 0, 0, 0))

            x = max(int(x), 0)
            y = max(int(y), 0)
            bw = max(int(bw), 0)
            bh = max(int(bh), 0)

            if bw == 0 or bh == 0:
                continue

            boxes.append((x, y, bw, bh))

        return boxes