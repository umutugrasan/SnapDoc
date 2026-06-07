"""4 köşe -> dik üstten görünüm (top-down warp).

`detector.order_points` ile TL/TR/BR/BL sırasına dizilmiş köşeleri alıp
hedef dikdörtgenin genişlik/yüksekliğini hesaplar, getPerspectiveTransform
ile homography matrisini bulur ve warpPerspective uygular.
"""
from __future__ import annotations

import cv2
import numpy as np


def _target_size(corners: np.ndarray) -> tuple[int, int]:
    tl, tr, br, bl = corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    W = int(round(max(width_top, width_bottom)))
    H = int(round(max(height_left, height_right)))
    return max(W, 1), max(H, 1)


def warp_to_top_down(
    image_bgr: np.ndarray, corners: np.ndarray
) -> np.ndarray:
    """Köşelerle tanımlı dörtgeni dik dikdörtgene dönüştür."""
    corners = corners.astype("float32")
    W, H = _target_size(corners)
    dst = np.array(
        [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32"
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image_bgr, M, (W, H), flags=cv2.INTER_CUBIC)
