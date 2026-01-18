from typing import Tuple
import cv2

Box = Tuple[int, int, int, int]  # (x, y, w, h)


def apply_blur(frame_bgr, box: Box, kernel_size: int = 35):
 
 
    if frame_bgr is None:
        return frame_bgr

    x, y, w, h = box
    h_img, w_img = frame_bgr.shape[:2]

    
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, w_img)
    y2 = min(y + h, h_img)

    if x2 <= x1 or y2 <= y1:
        return frame_bgr

   
    k = int(kernel_size)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    roi = frame_bgr[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (k, k), 0)

    frame_bgr[y1:y2, x1:x2] = blurred
    return frame_bgr