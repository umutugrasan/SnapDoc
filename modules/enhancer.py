"""Gölge/aydınlatma düzeltme + scanner görünümü.

Üç çıktı modu:
- "color":  gölgesiz, normalize edilmiş renkli görüntü
- "gray":   tek kanal, kontrast normalize
- "bw":     adaptif eşikleme ile siyah-beyaz scanner çıktısı (OCR için ideal)
"""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

Mode = Literal["color", "gray", "bw"]


def _estimate_background(gray: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Yumuşak gölge/arka plan tahmini (dilate + median blur)."""
    dilated = cv2.dilate(gray, np.ones((kernel_size, kernel_size), np.uint8))
    return cv2.medianBlur(dilated, 21)


def _remove_shadow_gray(gray: np.ndarray) -> np.ndarray:
    bg = _estimate_background(gray)
    diff = 255 - cv2.absdiff(gray, bg)
    return cv2.normalize(
        diff, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U,
    )


def _remove_shadow_color(image_bgr: np.ndarray) -> np.ndarray:
    """Her kanal için ayrı gölge çıkarımı (kağıdı beyaza, metni canlıya çek)."""
    channels = cv2.split(image_bgr)
    out = [_remove_shadow_gray(ch) for ch in channels]
    return cv2.merge(out)


def _sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype="float32")
    return cv2.filter2D(image, -1, kernel)


def enhance_scan(
    warped_bgr: np.ndarray,
    mode: Mode = "bw",
    sharpen: bool = True,
    block_size: int = 21,
    C: int = 12,
) -> np.ndarray:
    """Perspektif düzeltilmiş görüntüyü 'scanner' çıktısına dönüştür.

    block_size, C: adaptiveThreshold parametreleri (bw modunda).
    """
    if warped_bgr is None or warped_bgr.size == 0:
        raise ValueError("enhance_scan: boş görüntü")

    if mode == "color":
        out = _remove_shadow_color(warped_bgr)
        return _sharpen(out) if sharpen else out

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    norm = _remove_shadow_gray(gray)

    if mode == "gray":
        return _sharpen(norm) if sharpen else norm

    # mode == "bw"
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C,
    )
    return binary
