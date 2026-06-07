"""SnapDoc — Streamlit web arayüzü.

Çalıştır:  streamlit run app.py

Akış: yükle/çek -> köşe tespiti -> warp -> enhance -> OCR -> indir (PDF/Word/TXT).
Manuel köşe düzeltme (streamlit-drawable-canvas mevcutsa) etkinleştirilir.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# --- streamlit-drawable-canvas uyum yaması ---------------------------------
# Streamlit, image_to_url'ı streamlit.elements.image'ten
# streamlit.elements.lib.image_utils'a taşıdı VE imzasını değiştirdi
# (eski: (image, width:int, clamp, channels, output_format, image_id)).
# streamlit-drawable-canvas 0.9.x hâlâ eski imzayla çağırıyor.
# Bu yama, eski isim altına imzayı çeviren bir adapter bağlar.
def _install_canvas_compat() -> None:
    import streamlit.elements.image as _st_image
    if hasattr(_st_image, "image_to_url"):
        return
    try:
        from streamlit.elements.lib import image_utils as _img_utils
    except ImportError:
        return

    _new_fn = getattr(_img_utils, "image_to_url", None)
    if _new_fn is None:
        return

    def _build_layout_config(width):
        """`.width` özniteliğine sahip, kütüphanenin yeni imzasına uygun obje."""
        WidthClass = getattr(_img_utils, "WidthBehavior", None)
        if WidthClass is not None:
            # Dataclass / NamedTuple varyantlarını dene
            for ctor in (lambda: WidthClass(width=width), lambda: WidthClass(width)):
                try:
                    return ctor()
                except TypeError:
                    continue
        from types import SimpleNamespace
        return SimpleNamespace(width=width, use_container_width=False)

    import inspect
    _params = list(inspect.signature(_new_fn).parameters.keys())

    def _adapter(image, width, clamp=False, channels="RGB",
                 output_format="auto", image_id=""):
        layout = _build_layout_config(width)
        # Yeni daraltılmış imza: (image_data, layout_config, image_format)
        if len(_params) <= 3:
            return _new_fn(image, layout, output_format)
        # Geçiş dönemi imzası: tüm eski parametreleri de kabul ediyor
        return _new_fn(image, layout, clamp, channels, output_format, image_id)

    _st_image.image_to_url = _adapter


try:
    _install_canvas_compat()
except Exception:
    pass
# ---------------------------------------------------------------------------

from modules.detector import (
    detect_document_corners,
    draw_corners,
    fallback_full_frame,
    order_points,
)
from modules.enhancer import enhance_scan
from modules.exporter import (
    export_docx_multipage,
    export_pdf_multipage,
    export_txt,
)
from modules.ocr_engine import OCREngine
from modules.perspective import warp_to_top_down

st.set_page_config(page_title="SnapDoc", page_icon="📄", layout="wide")


# -------------------- yardımcılar --------------------

def _pil_to_bgr(pil: Image.Image) -> np.ndarray:
    arr = np.array(pil.convert("RGB"))
    return arr[:, :, ::-1].copy()


def _bgr_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(image[:, :, ::-1])


@st.cache_resource(show_spinner="OCR modeli yükleniyor (ilk seferde yavaş)…")
def get_ocr_engine(languages: tuple[str, ...]) -> OCREngine:
    return OCREngine(languages=list(languages), gpu=False)


def _numeric_corner_inputs(image_bgr: np.ndarray, idx: int) -> np.ndarray:
    """Canvas yedek planı: 4 köşeyi sayı inputlarıyla al."""
    h, w = image_bgr.shape[:2]
    st.caption("Köşe koordinatlarını piksel olarak gir (0,0 = sol-üst).")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Sol-Üst (TL)**")
        tl_x = st.number_input("TL x", 0, w - 1, 0, key=f"tlx_{idx}")
        tl_y = st.number_input("TL y", 0, h - 1, 0, key=f"tly_{idx}")
        st.markdown("**Sol-Alt (BL)**")
        bl_x = st.number_input("BL x", 0, w - 1, 0, key=f"blx_{idx}")
        bl_y = st.number_input("BL y", 0, h - 1, h - 1, key=f"bly_{idx}")
    with cB:
        st.markdown("**Sağ-Üst (TR)**")
        tr_x = st.number_input("TR x", 0, w - 1, w - 1, key=f"trx_{idx}")
        tr_y = st.number_input("TR y", 0, h - 1, 0, key=f"try_{idx}")
        st.markdown("**Sağ-Alt (BR)**")
        br_x = st.number_input("BR x", 0, w - 1, w - 1, key=f"brx_{idx}")
        br_y = st.number_input("BR y", 0, h - 1, h - 1, key=f"bry_{idx}")
    return np.array(
        [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]],
        dtype="float32",
    )


def _manual_corner_ui(image_bgr: np.ndarray, idx: int) -> np.ndarray | None:
    """Tıklanabilir canvas; başarısızsa sayı inputlu yedek."""
    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.info(
            "streamlit-drawable-canvas yok; köşeleri sayıyla girerek devam ediyoruz."
        )
        return _numeric_corner_inputs(image_bgr, idx)

    try:
        st.caption("Belgenin 4 köşesini sırayla tıkla (TL → TR → BR → BL).")
        disp = _bgr_to_pil(image_bgr)
        max_w = 700
        scale = min(1.0, max_w / disp.width)
        disp_w, disp_h = int(disp.width * scale), int(disp.height * scale)
        canvas_res = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_width=4, stroke_color="#00FF00",
            background_image=disp.resize((disp_w, disp_h)),
            update_streamlit=True,
            height=disp_h, width=disp_w,
            drawing_mode="point", point_display_radius=6,
            key=f"canvas_{idx}",
        )
    except Exception as e:
        st.warning(
            f"Kanvas yüklenemedi ({type(e).__name__}). "
            "Köşeleri sayıyla girmeye geçiyoruz."
        )
        return _numeric_corner_inputs(image_bgr, idx)

    if canvas_res is None or canvas_res.json_data is None:
        return None
    objs = canvas_res.json_data.get("objects", [])
    pts = [(o["left"], o["top"]) for o in objs if o.get("type") == "circle"]
    if len(pts) < 4:
        st.caption(f"({len(pts)}/4 köşe işaretlendi)")
        return None
    return np.array(pts[:4], dtype="float32") / scale


def process_one(
    image_bgr: np.ndarray,
    mode: str,
    manual_corners: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(preview_with_corners, warped, enhanced) döndür."""
    if manual_corners is not None:
        corners = order_points(manual_corners)
    else:
        corners = detect_document_corners(image_bgr)
        if corners is None:
            corners = fallback_full_frame(image_bgr)
    preview = draw_corners(image_bgr, corners)
    warped = warp_to_top_down(image_bgr, corners)
    enhanced = enhance_scan(warped, mode=mode)
    return preview, warped, enhanced


# -------------------- yan panel --------------------

st.sidebar.title("📄 SnapDoc")
st.sidebar.caption("Akıllı belge tarayıcı + OCR")

source = st.sidebar.radio(
    "Kaynak", ["Dosya yükle", "Kamera çek"], horizontal=True,
)
mode = st.sidebar.selectbox(
    "Mod", ["bw", "gray", "color"],
    format_func={"bw": "Siyah-Beyaz (scanner)", "gray": "Gri", "color": "Renkli"}.get,
)
langs = st.sidebar.multiselect(
    "OCR dilleri", ["tr", "en", "de", "fr", "ar", "ru"], default=["tr", "en"],
)
fmt = st.sidebar.selectbox(
    "Çıktı formatı", ["pdf", "docx", "txt"],
    format_func={"pdf": "Arama yapılabilir PDF", "docx": "Word (.docx)", "txt": "Düz metin"}.get,
)
manual_toggle = st.sidebar.checkbox("Manuel köşe düzeltme")
run_ocr = st.sidebar.checkbox("OCR çalıştır", value=True)


# -------------------- görüntü girişi --------------------

st.title("Belge tarama")
files = []
if source == "Dosya yükle":
    files = st.file_uploader(
        "Bir veya daha fazla belge fotoğrafı yükle",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )
else:
    snap = st.camera_input("Kamera ile yakala")
    if snap is not None:
        files = [snap]

if not files:
    st.info("Soldan kaynağı seç, ardından bir görüntü yükle veya kamerayla çek.")
    st.stop()

# -------------------- her sayfa için pipeline --------------------

pages_data: list[tuple[np.ndarray, list]] = []  # (enhanced, ocr_items)

ocr_engine = get_ocr_engine(tuple(langs)) if run_ocr and langs else None

for idx, f in enumerate(files, start=1):
    st.markdown(f"### Sayfa {idx}: `{getattr(f, 'name', 'kamera.jpg')}`")
    image_bgr = _pil_to_bgr(Image.open(f))

    manual_corners = None
    if manual_toggle:
        manual_corners = _manual_corner_ui(image_bgr, idx)

    preview, warped, enhanced = process_one(
        image_bgr, mode=mode, manual_corners=manual_corners,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.image(_bgr_to_pil(preview), caption="Orijinal + tespit", use_container_width=True)
    with c2:
        st.image(_bgr_to_pil(enhanced), caption="İyileştirilmiş çıktı", use_container_width=True)

    items = []
    if ocr_engine is not None:
        with st.spinner("OCR çalışıyor…"):
            items = ocr_engine.extract(enhanced)
        with st.expander(f"OCR metni ({len(items)} blok)"):
            st.text("\n".join(i.text for i in items))

    pages_data.append((enhanced, items))


# -------------------- indir butonları --------------------

st.markdown("---")
st.subheader("İndir")

out_buf = io.BytesIO()
tmp_path = Path(".snapdoc_tmp_out") / f"output.{fmt}"
tmp_path.parent.mkdir(exist_ok=True)

if fmt == "pdf":
    export_pdf_multipage(tmp_path, pages_data)
elif fmt == "docx":
    export_docx_multipage(tmp_path, pages_data)
else:
    # txt: tüm sayfaları birleştir
    from modules.exporter import _items_to_lines
    text_parts = []
    for i, (_, items) in enumerate(pages_data, 1):
        text_parts.append(f"--- Sayfa {i} ---")
        text_parts.extend(_items_to_lines(items))
    tmp_path.write_text("\n".join(text_parts), encoding="utf-8")

out_buf.write(tmp_path.read_bytes())
out_buf.seek(0)

mime = {"pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain"}[fmt]
st.download_button(
    f"⬇️ {fmt.upper()} indir",
    data=out_buf,
    file_name=f"snapdoc.{fmt}",
    mime=mime,
    use_container_width=True,
)
