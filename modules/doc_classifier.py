"""Belge tipi tahmini + yapılandırılmış alan eşleme.

Heuristic + regex tabanlı:
- "fatura"     : fatura no, tarih, toplam, kdv, satıcı VKN
- "makbuz"     : tarih, toplam, satıcı
- "kimlik"     : ad, soyad, TC kimlik, doğum tarihi
- "sözleşme"   : taraflar, tarih, imza
- "diğer"      : sınıflandırılamadı

Daha güçlü model için sonradan TF-IDF + sklearn LogisticRegression eklenebilir.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .ocr_engine import OCRItem


# Anahtar kelime sözlüğü (TR + EN). Skor: kaç anahtar kelime eşleşti?
KEYWORDS: dict[str, list[str]] = {
    "fatura": [
        "fatura", "invoice", "vkn", "tckn", "kdv", "ettn",
        "tutar", "toplam", "ödenecek", "tax invoice", "vat",
    ],
    "makbuz": [
        "makbuz", "receipt", "fiş", "kasa fişi", "ödeme", "yazar kasa",
    ],
    "kimlik": [
        "t.c. kimlik", "tc kimlik", "kimlik no", "republic of türkiye",
        "republic of turkey", "soyadı", "surname", "given name",
        "doğum tarihi", "date of birth",
    ],
    "sözleşme": [
        "sözleşme", "agreement", "contract", "taraflar", "between",
        "imzaları", "şartlar", "hükümleri",
    ],
}


@dataclass
class Classification:
    doc_type: str
    score: int
    fields: dict[str, str] = field(default_factory=dict)


def _full_text(items: Sequence[OCRItem]) -> str:
    return " ".join(i.text for i in items).lower()


# -------------------- alan çıkarımı --------------------

_RE_DATE = re.compile(
    r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b"
)
_RE_AMOUNT = re.compile(
    r"(\d{1,3}(?:[.\s]\d{3})*(?:[,\.]\d{2}))\s*(?:tl|try|₺|usd|eur|\$)?",
    re.IGNORECASE,
)
_RE_TCKN = re.compile(r"\b(\d{11})\b")
_RE_VKN = re.compile(r"\b(\d{10})\b")
_RE_INVOICE_NO = re.compile(
    r"(?:fatura\s*(?:no|numaras[ıi])|invoice\s*(?:no|number))\s*[:#]?\s*([A-Z0-9\-\/]{4,})",
    re.IGNORECASE,
)
_RE_KDV = re.compile(
    r"(?:kdv|vat)\s*(?:tutar[ıi])?\s*[:%]?\s*([0-9.,]+)",
    re.IGNORECASE,
)
_RE_TOTAL = re.compile(
    r"(?:genel\s*toplam|toplam|ödenecek|grand\s*total|total)\s*[:]?\s*"
    r"([0-9.,]+)",
    re.IGNORECASE,
)


def _extract_invoice_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    if m := _RE_INVOICE_NO.search(text):
        fields["fatura_no"] = m.group(1)
    if m := _RE_DATE.search(text):
        fields["tarih"] = m.group(1)
    if m := _RE_TOTAL.search(text):
        fields["toplam"] = m.group(1)
    if m := _RE_KDV.search(text):
        fields["kdv"] = m.group(1)
    if m := _RE_VKN.search(text):
        fields["vkn"] = m.group(1)
    return fields


def _extract_id_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    if m := _RE_TCKN.search(text):
        fields["tc_kimlik"] = m.group(1)
    if m := _RE_DATE.search(text):
        fields["dogum_tarihi"] = m.group(1)
    return fields


def _extract_receipt_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    if m := _RE_DATE.search(text):
        fields["tarih"] = m.group(1)
    if m := _RE_AMOUNT.search(text):
        fields["tutar"] = m.group(1)
    return fields


# -------------------- ana giriş --------------------

def classify(items: Sequence[OCRItem]) -> Classification:
    text = _full_text(items)
    scores = {t: sum(1 for k in kws if k in text) for t, kws in KEYWORDS.items()}
    best_type, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return Classification(doc_type="diğer", score=0)

    if best_type == "fatura":
        fields = _extract_invoice_fields(text)
    elif best_type == "kimlik":
        fields = _extract_id_fields(text)
    elif best_type == "makbuz":
        fields = _extract_receipt_fields(text)
    else:
        fields = {}
        if m := _RE_DATE.search(text):
            fields["tarih"] = m.group(1)

    return Classification(doc_type=best_type, score=best_score, fields=fields)
