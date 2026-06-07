"""Belge kenarı ve 4 köşe tespiti.

Pipeline: gri -> Gauss blur -> Canny -> morfolojik kapama -> kontur ->
en büyük dörtgen yaklaştırma -> köşeleri TL/TR/BR/BL sırasına diz.
"""
from __future__ import annotations

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """4 noktayı TL, TR, BR, BL sırasına diz."""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL: x+y en küçük
    rect[2] = pts[np.argmax(s)]  # BR: x+y en büyük

    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # TR: x-y en küçük (y-x büyük)
    rect[3] = pts[np.argmax(diff)]  # BL: x-y en büyük
    return rect


def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def detect_document_corners(
    image_bgr: np.ndarray,
    min_area_ratio: float = 0.1,
) -> np.ndarray | None:
    """4 köşeyi (TL, TR, BR, BL) döndür, bulunamazsa None.

    min_area_ratio: dörtgenin görüntüye oranı bu eşikten küçükse atla.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    h, w = image_bgr.shape[:2]
    img_area = h * w
    edges = _preprocess(image_bgr)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * min_area_ratio:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_points(approx)
    return None


def fallback_full_frame(image_bgr: np.ndarray) -> np.ndarray:
    """Tespit başarısızsa tüm çerçeveyi köşe olarak döndür."""
    h, w = image_bgr.shape[:2]
    return np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32"
    )


def draw_corners(
    image_bgr: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Tespit edilen dörtgeni görüntüye çiz (önizleme için)."""
    out = image_bgr.copy()
    pts = corners.astype(int).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    for (x, y) in corners.astype(int):
        cv2.circle(out, (int(x), int(y)), 8, color, -1)
    return out
