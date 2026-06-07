"""Çizgisel tablo tespiti ve Excel'e aktarım.

Klasik CV yaklaşımı:
1. Adaptif eşikleme + tersleme (siyah=metin/çizgi, beyaz=arka plan)
2. Yatay ve dikey morfolojik kernel ile çizgileri ayır
3. Çizgileri birleştir -> tablo maskesi
4. Bağlı bileşen ile hücreleri bul, OCR ile içeriğini oku
5. openpyxl ile xlsx üret
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .ocr_engine import OCREngine


def _binarize(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    bin_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10,
    )
    return bin_inv


def detect_table_lines(image: np.ndarray) -> np.ndarray:
    """Yatay+dikey çizgi maskesini döndür (0/255)."""
    bin_inv = _binarize(image)
    h, w = bin_inv.shape

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 10), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 30, 10)))

    horiz = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
    vert = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, vert_kernel, iterations=2)
    return cv2.addWeighted(horiz, 0.5, vert, 0.5, 0)


def _find_cells(grid_mask: np.ndarray, min_area: int = 400) -> list[tuple[int, int, int, int]]:
    """Tablo maskesinden hücre dikdörtgenlerini çıkar."""
    inv = cv2.bitwise_not(grid_mask)
    num, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    cells = []
    H, W = grid_mask.shape
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if w >= W * 0.98 and h >= H * 0.98:
            continue  # tüm sayfa
        cells.append((int(x), int(y), int(w), int(h)))
    return cells


def _group_into_grid(
    cells: list[tuple[int, int, int, int]], row_tol: int = 15,
) -> list[list[tuple[int, int, int, int]]]:
    """Hücreleri satırlara grupla (y-merkezine göre)."""
    if not cells:
        return []
    cells_sorted = sorted(cells, key=lambda b: (b[1] + b[3] / 2, b[0]))
    rows: list[list[tuple[int, int, int, int]]] = []
    current: list[tuple[int, int, int, int]] = []
    last_yc = None
    for cell in cells_sorted:
        yc = cell[1] + cell[3] / 2
        if last_yc is None or abs(yc - last_yc) <= row_tol:
            current.append(cell)
        else:
            rows.append(sorted(current, key=lambda c: c[0]))
            current = [cell]
        last_yc = yc
    if current:
        rows.append(sorted(current, key=lambda c: c[0]))
    return rows


def extract_table(
    image: np.ndarray, ocr_engine: OCREngine | None = None,
) -> list[list[str]]:
    """Tabloyu satır-sütun matrisine çevir (her hücre OCR ile okunur)."""
    grid = detect_table_lines(image)
    cells = _find_cells(grid)
    rows = _group_into_grid(cells)
    if not rows:
        return []

    engine = ocr_engine or OCREngine()
    matrix: list[list[str]] = []
    for row in rows:
        row_texts: list[str] = []
        for (x, y, w, h) in row:
            roi = image[y:y + h, x:x + w]
            text = engine.extract_plain_text(roi, sep=" ").strip()
            row_texts.append(text)
        matrix.append(row_texts)
    return matrix


def save_table_xlsx(matrix: list[list[str]], path: Path) -> None:
    """Tablo matrisini .xlsx olarak kaydet."""
    from openpyxl import Workbook

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "SnapDoc"
    for row in matrix:
        ws.append(row)
    wb.save(str(path))
