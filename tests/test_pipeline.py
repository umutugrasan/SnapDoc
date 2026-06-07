"""Hızlı smoke testleri — pytest gerektirmez.

Çalıştır:  python tests/test_pipeline.py

Sentetik bir 'belge' (beyaz dikdörtgen, koyu arka plan, hafif eğri) üretip
detector -> perspective -> enhancer zincirinin çalıştığını doğrular.

Bağımlılıklar: numpy, opencv-python.
EasyOCR / Streamlit / ReportLab burada test edilmez.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.detector import detect_document_corners, order_points  # noqa: E402
from modules.enhancer import enhance_scan  # noqa: E402
from modules.perspective import warp_to_top_down  # noqa: E402


def make_synthetic_document(size: int = 600) -> np.ndarray:
    """Koyu arka plan üzerinde beyaz, hafif döndürülmüş bir 'belge' üret."""
    img = np.full((size, size, 3), 30, dtype=np.uint8)  # koyu arka plan
    page = np.full((400, 300, 3), 240, dtype=np.uint8)  # beyaz sayfa
    # üzerine sahte metin satırları
    for y in range(40, 360, 30):
        cv2.line(page, (20, y), (280, y), (40, 40, 40), 2)

    # döndür ve büyük çerçeveye yerleştir
    M = cv2.getRotationMatrix2D((150, 200), 12.0, 1.0)
    page_rot = cv2.warpAffine(page, M, (300, 400), borderValue=(30, 30, 30))
    x0, y0 = (size - 300) // 2, (size - 400) // 2
    img[y0:y0 + 400, x0:x0 + 300] = page_rot
    return img


def test_order_points() -> None:
    raw = np.array([[100, 100], [10, 200], [10, 10], [200, 100]], dtype="float32")
    ordered = order_points(raw)
    assert ordered.shape == (4, 2)
    # TL en küçük toplam, BR en büyük
    assert ordered[0].sum() <= ordered[2].sum()
    print("  ✓ order_points")


def test_detector_finds_quad() -> None:
    img = make_synthetic_document()
    corners = detect_document_corners(img)
    assert corners is not None, "köşeler tespit edilemedi"
    assert corners.shape == (4, 2)
    print("  ✓ detect_document_corners")


def test_full_pipeline() -> None:
    img = make_synthetic_document()
    corners = detect_document_corners(img)
    assert corners is not None
    warped = warp_to_top_down(img, corners)
    assert warped.ndim == 3 and warped.shape[2] == 3
    assert warped.shape[0] > 50 and warped.shape[1] > 50

    enhanced_bw = enhance_scan(warped, mode="bw")
    assert enhanced_bw.ndim == 2
    # Siyah-beyaz olmalı (sadece 0 ve 255)
    uniq = np.unique(enhanced_bw)
    assert set(uniq.tolist()).issubset({0, 255})

    enhanced_gray = enhance_scan(warped, mode="gray")
    assert enhanced_gray.ndim == 2

    enhanced_color = enhance_scan(warped, mode="color")
    assert enhanced_color.ndim == 3

    print("  ✓ full pipeline (bw/gray/color)")


def main() -> int:
    print("SnapDoc smoke testleri çalışıyor…")
    test_order_points()
    test_detector_finds_quad()
    test_full_pipeline()
    print("Tüm testler geçti.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
