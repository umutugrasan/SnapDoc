"""Çıktı üretimi: arama yapılabilir PDF, Word (.docx), düz metin (.txt).

PDF stratejisi:
- ReportLab canvas üzerine **sayfa boyutunda görüntü** çizilir.
- OCR'dan gelen her metin bloğu, görüntünün üstünde **görünmez (render mode 3)**
  bir text katmanı olarak yerleştirilir. Böylece görsel kalite korunur ama
  Acrobat / Foxit / Chrome PDF'te arama ve kopyalama çalışır.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .ocr_engine import OCRItem


# -------------------- yardımcılar --------------------

def _ndarray_to_pil(image: np.ndarray) -> Image.Image:
    """BGR / Gri / Binary numpy görüntüsünü PIL.Image (RGB)'ye çevir."""
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    # OpenCV BGR -> RGB
    return Image.fromarray(image[:, :, ::-1]).convert("RGB")


# -------------------- PDF --------------------

def export_pdf(
    path: Path,
    image: np.ndarray,
    ocr_items: Sequence[OCRItem] = (),
    dpi: int = 200,
) -> None:
    """Tek sayfalı arama yapılabilir PDF üretir."""
    export_pdf_multipage(path, [(image, ocr_items)], dpi=dpi)


def export_pdf_multipage(
    path: Path,
    pages: Iterable[tuple[np.ndarray, Sequence[OCRItem]]],
    dpi: int = 200,
) -> None:
    """Birden fazla (görüntü, ocr_items) sayfasından tek PDF üret."""
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(path))
    for image, items in pages:
        pil = _ndarray_to_pil(image)
        # piksel -> punto: 1 inç = 72 punto
        w_pt = pil.width * 72.0 / dpi
        h_pt = pil.height * 72.0 / dpi
        c.setPageSize((w_pt, h_pt))

        # Görüntü katmanı
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        c.drawImage(ImageReader(buf), 0, 0, width=w_pt, height=h_pt)

        # Görünmez OCR text katmanı (render mode 3 = invisible)
        _draw_invisible_text_layer(c, items, pil.width, pil.height, w_pt, h_pt)
        c.showPage()
    c.save()


def _draw_invisible_text_layer(
    c, items: Sequence[OCRItem], px_w: int, px_h: int, pt_w: float, pt_h: float,
) -> None:
    """OCR sonuçlarını PDF üstüne görünmez metin olarak yerleştir."""
    if not items:
        return
    sx = pt_w / max(px_w, 1)
    sy = pt_h / max(px_h, 1)

    text = c.beginText()
    text.setTextRenderMode(3)  # invisible
    for item in items:
        x, y, w, h = item.bbox
        if not item.text or w <= 0 or h <= 0:
            continue
        # PDF koordinat sistemi sol-alt başlangıçlı
        x_pt = x * sx
        y_pt = pt_h - (y + h) * sy
        font_size = max(h * sy * 0.85, 1.0)
        text.setFont("Helvetica", font_size)
        text.setTextOrigin(x_pt, y_pt)
        text.textOut(item.text)
    c.drawText(text)


# -------------------- Word --------------------

def export_docx(
    path: Path,
    image: np.ndarray,
    ocr_items: Sequence[OCRItem] = (),
    embed_image: bool = True,
) -> None:
    export_docx_multipage(path, [(image, ocr_items)], embed_image=embed_image)


def export_docx_multipage(
    path: Path,
    pages: Iterable[tuple[np.ndarray, Sequence[OCRItem]]],
    embed_image: bool = True,
) -> None:
    """Çoklu sayfa: her sayfa Word'de yeni section. Görüntü + OCR metin."""
    from docx import Document
    from docx.shared import Inches

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()
    first = True
    for image, items in pages:
        if not first:
            doc.add_page_break()
        first = False

        if embed_image:
            pil = _ndarray_to_pil(image)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            buf.seek(0)
            doc.add_picture(buf, width=Inches(6.0))

        for line in _items_to_lines(items):
            doc.add_paragraph(line)

    doc.save(str(path))


def _items_to_lines(items: Sequence[OCRItem]) -> list[str]:
    """OCR sonuçlarını okuma sırasıyla satırlara böl (üst->alt, sol->sağ)."""
    if not items:
        return []
    # y-merkezine göre satır gruplama
    enriched = []
    for it in items:
        x, y, w, h = it.bbox
        enriched.append((y + h / 2, x, it))
    enriched.sort(key=lambda t: (t[0], t[1]))

    lines: list[list[OCRItem]] = []
    current: list[OCRItem] = []
    last_y = None
    for yc, _x, it in enriched:
        if last_y is None or abs(yc - last_y) <= max(it.bbox[3], 10) * 0.7:
            current.append(it)
        else:
            lines.append(current)
            current = [it]
        last_y = yc
    if current:
        lines.append(current)

    out = []
    for line in lines:
        line.sort(key=lambda i: i.bbox[0])
        out.append(" ".join(i.text for i in line))
    return out


# -------------------- TXT --------------------

def export_txt(
    path: Path,
    image: np.ndarray,  # imzayı tutmak için (kullanılmıyor)
    ocr_items: Sequence[OCRItem] = (),
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _items_to_lines(ocr_items)
    path.write_text("\n".join(lines), encoding="utf-8")


# -------------------- birleşik dispatcher --------------------

def export(
    path: Path,
    image: np.ndarray,
    ocr_items: Sequence[OCRItem] = (),
    fmt: str | None = None,
) -> None:
    """Tek sayfalı dışa aktarımın birleşik girişi.

    fmt None ise dosya uzantısından çıkarılır.
    """
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt == "pdf":
        export_pdf(path, image, ocr_items)
    elif fmt in {"docx", "doc"}:
        export_docx(path, image, ocr_items)
    elif fmt == "txt":
        export_txt(path, image, ocr_items)
    else:
        raise ValueError(f"Bilinmeyen format: {fmt}")
