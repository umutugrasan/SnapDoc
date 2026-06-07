# SnapDoc — Akıllı Belge Tarayıcı + OCR

Telefonla / webcam'le çekilmiş eğri-açılı bir belge fotoğrafını **scanner kalitesinde** düz sayfaya dönüştüren, OCR ile metni çıkarıp arama yapılabilir **PDF / Word / Excel** olarak dışa aktaran uçtan uca bir Python projesi.

## Hızlı Başlangıç

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Web sürümü
streamlit run app.py

# Komut satırı
python snapdoc_cli.py --input samples\fatura.jpg --output outputs\fatura.pdf --format pdf
```

## Mimari

```
detector → perspective → enhancer → ocr_engine → exporter
```

Her modül bağımsız test edilebilir; sadece NumPy array alır, NumPy array veya yapılandırılmış dict döner.

## Modüller

| Modül | Sorumluluk |
|-------|------------|
| `detector.py` | Canny + kontur + 4 köşe yaklaştırma |
| `perspective.py` | `getPerspectiveTransform` + `warpPerspective` |
| `enhancer.py` | Gölge kaldırma + adaptif eşikleme |
| `ocr_engine.py` | EasyOCR (TR+EN) sarmalayıcı |
| `exporter.py` | Arama yapılabilir PDF / Word / TXT |
| `table_extractor.py` | (Bonus) Tablo → Excel |
| `doc_classifier.py` | (Bonus) Belge tipi tahmini |

Tam teknik dokümantasyon: [SnapDoc_Dokumantasyonu.md](SnapDoc_Dokumantasyonu.md)
