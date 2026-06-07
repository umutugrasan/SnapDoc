"""EasyOCR sarmalayıcı.

EasyOCR Reader örneği pahalı (model yükleme) — sınıf üzerinden tek seferlik
yükleyip birden fazla görüntüde tekrar kullanılır.

Dönüş: List[OCRItem] = [(bbox=(x,y,w,h), text, confidence)].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class OCRItem:
    """Tek bir OCR sonucu."""
    bbox: tuple[int, int, int, int]   # x, y, w, h (sol-üst köşe + boyut)
    text: str
    confidence: float

    @property
    def polygon(self) -> list[tuple[int, int]]:
        x, y, w, h = self.bbox
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _to_xywh(quad) -> tuple[int, int, int, int]:
    """EasyOCR dörtgenini (4 nokta) (x, y, w, h) sınırlayıcı kutuya çevir."""
    pts = np.asarray(quad, dtype="float32").reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return (
        int(round(x_min)),
        int(round(y_min)),
        int(round(x_max - x_min)),
        int(round(y_max - y_min)),
    )


class OCREngine:
    """Lazy-loaded EasyOCR sarmalayıcısı.

    >>> engine = OCREngine(languages=["tr", "en"])
    >>> items = engine.extract(scan_image)
    >>> "\\n".join(i.text for i in items)
    """

    def __init__(
        self,
        languages: Iterable[str] = ("tr", "en"),
        gpu: bool = False,
        min_confidence: float = 0.0,
    ) -> None:
        self.languages = list(languages)
        self.gpu = gpu
        self.min_confidence = float(min_confidence)
        self._reader = None  # easyocr.Reader (lazy)

    @property
    def reader(self):
        if self._reader is None:
            import easyocr  # ağır import — sadece gerektiğinde
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        return self._reader

    def extract(self, image: np.ndarray) -> list[OCRItem]:
        if image is None or image.size == 0:
            return []
        raw = self.reader.readtext(image)
        items: list[OCRItem] = []
        for quad, text, conf in raw:
            conf = float(conf)
            if conf < self.min_confidence:
                continue
            text = (text or "").strip()
            if not text:
                continue
            items.append(OCRItem(bbox=_to_xywh(quad), text=text, confidence=conf))
        return items

    def extract_plain_text(self, image: np.ndarray, sep: str = "\n") -> str:
        return sep.join(item.text for item in self.extract(image))
