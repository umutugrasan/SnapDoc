"""SnapDoc komut satırı arayüzü.

Faz 1: tek görüntü -> köşe tespiti -> warp -> enhance -> kaydet.
Faz 2 (sonra): + OCR + PDF/Word/TXT çıktı.

Kullanım:
    python snapdoc_cli.py --input samples/fatura.jpg --output outputs/fatura.png
    python snapdoc_cli.py --input samples/ --output outputs/ --mode bw
    python snapdoc_cli.py --input samples/fatura.jpg --output outputs/fatura.pdf \
        --format pdf --lang tr en
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from modules.detector import (
    detect_document_corners,
    draw_corners,
    fallback_full_frame,
)
from modules.enhancer import enhance_scan
from modules.perspective import warp_to_top_down

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def process_image(
    image_bgr: np.ndarray,
    mode: str = "bw",
    debug_dir: Path | None = None,
    stem: str = "doc",
) -> np.ndarray:
    """Tek görüntü için çekirdek pipeline."""
    corners = detect_document_corners(image_bgr)
    if corners is None:
        print(f"  ! köşe tespit edilemedi, tüm çerçeve kullanılacak: {stem}")
        corners = fallback_full_frame(image_bgr)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        preview = draw_corners(image_bgr, corners)
        cv2.imwrite(str(debug_dir / f"{stem}_corners.jpg"), preview)

    warped = warp_to_top_down(image_bgr, corners)
    enhanced = enhance_scan(warped, mode=mode)
    return enhanced


def _iter_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMG_EXT)
    if input_path.is_file():
        return [input_path]
    raise FileNotFoundError(input_path)


def _resolve_output(out_path: Path, src: Path, multi: bool, fmt_ext: str) -> Path:
    """Çıktı yolunu çöz: tek dosya ya da klasör hedefi."""
    if multi or out_path.is_dir() or out_path.suffix == "":
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path / f"{src.stem}{fmt_ext}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="snapdoc",
        description="SnapDoc: belge fotoğrafı -> scanner çıktısı + OCR.",
    )
    p.add_argument("--input", "-i", required=True, type=Path,
                   help="Görüntü dosyası veya klasör")
    p.add_argument("--output", "-o", required=True, type=Path,
                   help="Çıktı dosyası veya klasör")
    p.add_argument("--mode", choices=["color", "gray", "bw"], default="bw",
                   help="Enhance modu (varsayılan: bw)")
    p.add_argument("--format", choices=["png", "jpg", "pdf", "docx", "txt"],
                   default="png", help="Çıktı formatı")
    p.add_argument("--lang", nargs="+", default=["tr", "en"],
                   help="OCR dilleri (örn: tr en)")
    p.add_argument("--debug", action="store_true",
                   help="Köşe önizleme görüntüleri kaydet")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = _iter_inputs(args.input)
    if not sources:
        print(f"Girdi boş: {args.input}", file=sys.stderr)
        return 1

    multi = len(sources) > 1
    fmt_ext = f".{args.format}"
    debug_dir = (args.output if args.output.is_dir() else args.output.parent) / "_debug"

    # OCR + dışa aktarım gerekli mi? (lazy import)
    need_ocr = args.format in {"pdf", "docx", "txt"}
    ocr_reader = None
    exporter = None
    if need_ocr:
        from modules.exporter import export
        from modules.ocr_engine import OCREngine
        ocr_reader = OCREngine(languages=args.lang)
        exporter = export

    for src in sources:
        print(f"[+] {src.name}")
        image = cv2.imread(str(src))
        if image is None:
            print(f"  ! okunamadı: {src}", file=sys.stderr)
            continue

        result = process_image(
            image, mode=args.mode,
            debug_dir=debug_dir if args.debug else None,
            stem=src.stem,
        )

        target = _resolve_output(args.output, src, multi, fmt_ext)

        if args.format in {"png", "jpg"}:
            cv2.imwrite(str(target), result)
        else:
            ocr_items = ocr_reader.extract(result)
            exporter(
                target, image=result, ocr_items=ocr_items, fmt=args.format,
            )
        print(f"    -> {target}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
