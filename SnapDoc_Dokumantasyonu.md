# SNAPDOC: AKILLI BELGE TARAYICI + OCR PLATFORMU

*Python, OpenCV, EasyOCR ve Streamlit Tabanlı Portfolyo Projesi Teknik Dokümantasyonu*

> **Portfolyo Vizyonu:** Bu proje, telefonla / webcam'le çekilmiş eğri, gölgeli, açılı bir belge fotoğrafını **scanner görüntüsü kalitesinde** düz bir sayfaya dönüştürür; üzerindeki metni çıkarır ve arama yapılabilir PDF / Word / Excel olarak dışa aktarır. Klasik bilgisayarlı görü (kenar tespit, perspektif düzeltme, görüntü iyileştirme) ile modern OCR'ı birleştiren, uçtan uca bir veri çıkarım hattıdır.

---

## 1. Proje Tanımı ve Temel İşlevler

**SnapDoc**, kullanıcının kameradan canlı yakaladığı veya yüklediği eğri-açılı bir kağıt belgenin köşelerini otomatik tespit eder, **homography** ile düzeltir, gölge ve aydınlatma sorunlarını giderir, ardından metni **OCR** ile çıkarır. Tek bir belge kadar **çok sayfalı** belgeleri tek PDF'e birleştirebilir; fatura/makbuz gibi yapılandırılmış belgelerde **alan eşleme** yaparak doğrudan Excel'e atar.

**Tek cümleyle:** Kameradaki kağıt → 5 saniyede arama yapılabilir PDF/Word.

---

## 2. Uçtan Uca İş Akışı

```
[1] Görüntü Yakalama          (webcam akışı / dosya yükleme)
        │
        ▼
[2] Belge Kenarı Tespiti       (Canny + kontur + dörtgen yaklaştırma)
        │
        ▼
[3] Köşe Sıralama              (TL → TR → BR → BL)
        │
        ▼
[4] Perspektif Düzeltme        (cv2.getPerspectiveTransform + warpPerspective)
        │
        ▼
[5] Görüntü İyileştirme        (gölge kaldırma + adaptif eşikleme + keskinleştirme)
        │
        ▼
[6] OCR                        (EasyOCR / Tesseract — TR + EN)
        │
        ▼
[7] Yapılandırma               (metin blokları, tablolar, alan eşleme)
        │
        ▼
[8] Çıktı                       (arama yapılabilir PDF / Word / Excel)
```

---

## 3. Detaylı Özellik Listesi

### 3.1 Çekirdek Özellikler

| Özellik | Mantık (CV) | Çıktı |
|---------|-------------|-------|
| **Otomatik Kenar Tespiti** | Gri tonlama → Gauss bulanıklaştırma → Canny → en büyük dörtgen kontur | 4 köşe koordinatı |
| **Manuel Köşe Düzeltme** | Tespit yanlışsa kullanıcı 4 köşeyi sürükleyebilir (Streamlit canvas) | Düzeltilmiş köşe |
| **Perspektif Düzeltme** | `cv2.getPerspectiveTransform` ile 4 köşe → dikdörtgen homography | Üstten-dik görünüm |
| **Gölge ve Aydınlatma Düzeltme** | Morfolojik kapama ile arkaplan tahmini, sonra subtraction | Eşit aydınlatılmış görüntü |
| **Akıllı Eşikleme** | Sauvola / Adaptive Gaussian threshold ile siyah-beyaz "scanner" görünümü | İkili görüntü |
| **OCR (Tr + En)** | EasyOCR tabanlı çoklu dil tanıma | Konumlu metin bloğu |
| **Arama Yapılabilir PDF** | Görüntü + OCR text layer ReportLab ile birleşir | Tek dosya PDF |
| **Word (.docx) İhracatı** | python-docx ile başlık + paragraf yapısı korunur | Düzenlenebilir Word |
| **Çoklu Sayfa Birleştirme** | Arka arkaya yakalanan sayfaları tek PDF'te sırala | Çok sayfalı PDF |

### 3.2 İleri Özellikler (Bonus)

| Özellik | Açıklama |
|---------|----------|
| **Canlı Çerçeve Önizleme** | Webcam akışında belge çerçevesi gerçek zamanlı yeşil çizgiyle vurgulanır; kararlı dururken otomatik yakalama |
| **Tablo Tanıma → Excel** | Belgedeki çizgisel/gridsiz tablolar tespit edilip hücre eşlemesiyle .xlsx'e aktarılır |
| **Belge Tipi Sınıflandırma** | "Fatura", "Makbuz", "Kimlik", "Sözleşme" gibi tipler basit bir sınıflandırıcıyla etiketlenir |
| **Yapılandırılmış Alan Çıkarımı** | Faturada *toplam, tarih, KDV, satıcı*; kimlikte *ad, TC, doğum tarihi* — regex + konum heuristic ile alan eşleme |
| **Çoklu Dil OCR** | TR + EN + AR + RU... ek dil paketleri yüklenebilir |
| **Toplu İşleme (Batch)** | Klasördeki tüm fotoğrafları aynı anda işle, tek raporlu Excel'e aktar |

---

## 4. Teknolojik Altyapı ve Kütüphaneler

| Kütüphane | Rol |
|-----------|-----|
| **OpenCV (cv2)** | Görüntü işlemenin tamamı: kenar, kontur, perspektif, morfoloji, eşikleme |
| **NumPy** | Görüntü matrisleri ve köşe vektör hesapları |
| **EasyOCR** | Pretrained çoklu dil OCR; CPU'da makul, GPU'da hızlı |
| **Tesseract (opsiyonel)** | Hafif alternatif OCR motoru; LSTM tabanlı |
| **ReportLab** | Görüntü + arama yapılabilir text katmanı ile PDF üretimi |
| **python-docx** | Word çıktısı |
| **openpyxl** | Excel çıktısı (tablo ve alan eşleme) |
| **Streamlit + streamlit-webrtc** | Web UI; canlı webcam akışı + dosya yükleme |
| **Pillow (PIL)** | Format dönüşümleri, PDF'e gömme |

---

## 5. Yazılım Mimarisi ve Modüller

```
SnapDoc/
├── app.py                      # Streamlit web sürümü
├── snapdoc_cli.py              # Komut satırı / klasör toplu işleme
│
├── modules/
│   ├── detector.py             # Belge kenarı + köşe tespiti
│   ├── perspective.py          # Köşe sıralama + warpPerspective
│   ├── enhancer.py             # Gölge kaldırma, eşikleme, keskinleştirme
│   ├── ocr_engine.py           # EasyOCR/Tesseract sarmalayıcısı
│   ├── exporter.py             # PDF/Word/Excel çıktı üretimi
│   ├── table_extractor.py      # (Bonus) Tablo → Excel
│   └── doc_classifier.py       # (Bonus) Belge tipi tahmini
│
├── samples/                    # Test belgeleri (fatura, sayfa, makbuz)
├── outputs/                    # Üretilen PDF/Word/Excel
├── requirements.txt
└── README.md
```

**Modüllerin sorumluluk sınırı:** `detector → perspective → enhancer → ocr_engine → exporter` ardışık pipeline'dır. Her modül bağımsız test edilebilir (her biri sadece NumPy array alır, NumPy array veya structured dict döner).

---

## 6. Algoritma Çekirdeği (Pseudo + Python)

### 6.1 Belge Köşesi Tespiti

```python
import cv2
import numpy as np

def detect_document_corners(image_bgr):
    """Belgenin 4 köşesini (TL, TR, BR, BL sırasıyla) döndürür."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    # Morfolojik kapama ile kenarları kalınlaştır
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_points(approx.reshape(4, 2))
    return None  # tespit edilemedi
```

### 6.2 Köşe Sıralama (TL → TR → BR → BL)

```python
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)        # x + y küçük → TL, büyük → BR
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)  # x - y küçük → TR, büyük → BL
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
```

### 6.3 Perspektif Düzeltme

```python
def warp_to_top_down(image, corners):
    (tl, tr, br, bl) = corners
    wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
    W, H = int(max(wA, wB)), int(max(hA, hB))

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners.astype("float32"), dst)
    return cv2.warpPerspective(image, M, (W, H))
```

### 6.4 Gölge Kaldırma + Scanner Görünümü

```python
def enhance_scan(warped_bgr):
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Arka plan tahmini: büyük morfolojik kapama → yumuşak gölge
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)

    # 2) Normalize: piksel / arka plan, kontrastı geri yükle
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 3) Adaptif eşikleme — scanner görünümü
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 12
    )
    return binary
```

### 6.5 OCR ve Çıktı

```python
import easyocr

def extract_text(image, lang=["tr", "en"]):
    reader = easyocr.Reader(lang, gpu=False)
    result = reader.readtext(image)
    # [(bbox, text, confidence), ...]
    return result
```

PDF/Word çıktısı `exporter.py` içinde **görüntüyü** sayfaya, **OCR text'i** üstüne görünmez katmanda yerleştirir; böylece Acrobat'ta arama / kopyalama çalışır.

---

## 7. Kullanıcı Arayüzü (Streamlit)

Yan panel:
- **Kaynak**: kamera / dosya yükle / klasör (batch)
- **Dil**: TR + EN (varsayılan), AR, RU eklenebilir
- **Mod**: Renkli / Gri / Siyah-Beyaz scanner
- **Çıktı formatı**: PDF (önerilir), Word, Excel (tablo varsa), Düz metin
- **Manuel köşe düzeltme** anahtarı

Ana panel:
- Sol: orijinal görüntü + tespit edilen dörtgen
- Sağ: düzeltilmiş + iyileştirilmiş çıktı
- Alt: OCR sonucu metin önizleme + "PDF İndir" düğmesi

---

## 8. Portfolyoda Öne Çıkaran Hususlar

- **Uçtan uca veri ürünü**: Algı → düzeltme → tanıma → yapılandırılmış çıktı (PDF/Word/Excel) — her aşaması test edilebilir, modüler.
- **Klasik CV ustalığı**: Canny, kontur, morfoloji, homography — hiçbir yerde "hazır model çağırma" değil, **matematiği bilinerek yazılmış pipeline**.
- **OCR doğruluk optimizasyonu**: Eşikleme parametrelerinin OCR doğruluğuna etkisi ölçülebilir; doğruluk grafiği README'ye eklenir.
- **Gerçek dünya senaryosu**: Fatura otomasyonu, sözleşme arşivlemesi, kimlik doğrulama gibi somut iş senaryoları.
- **Genişletilebilirlik**: Aynı pipeline'ı LLM ile birleştirerek "fatura → muhasebe kaydı" gibi otomasyonlara çevirmek mümkün.

---

## 9. Kurulum ve Çalıştırma Adımları

```powershell
# 1) Python 3.11 ile sanal ortam (MediaPipe/EasyOCR uyumlu)
py -3.11 -m venv venv
venv\Scripts\activate

# 2) Bağımlılıklar
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Web sürümü
streamlit run app.py

# 4) Komut satırı / toplu işlem
python snapdoc_cli.py --input ./samples --output ./outputs --format pdf
```

**requirements.txt iskeleti**

```
numpy==1.26.4
opencv-python==4.10.0.84
easyocr==1.7.2
streamlit>=1.30,<2.0
streamlit-webrtc>=0.47.0
av>=11.0.0
reportlab>=4.0.0
python-docx>=1.1.0
openpyxl>=3.1.0
Pillow>=10.0.0
```

---

## 10. Geliştirme Yol Haritası (Faz Faz)

| Faz | Çıktı | Süre |
|-----|-------|------|
| **Faz 1 — Çekirdek pipeline** | Tek fotoğraf → düz, iyileştirilmiş çıktı (CLI) | 2 gün |
| **Faz 2 — OCR + PDF** | Düz çıktı + arama yapılabilir PDF | 1 gün |
| **Faz 3 — Streamlit UI** | Dosya yükleme + canlı webcam + indir butonu | 1 gün |
| **Faz 4 — Çoklu sayfa + Word** | Birden fazla sayfa + .docx çıktı | 1 gün |
| **Faz 5 — Manuel köşe düzeltme** | Kullanıcı 4 köşeyi sürükleyebilsin | 1 gün |
| **Faz 6 — Bonus** | Tablo → Excel, fatura alan çıkarımı | 2-3 gün |

**Toplam:** ~1.5 hafta için sergilenebilir kalitede MVP.

---

## 11. CV / Portfolyo Notu

SnapDoc, sadece bir OCR sarmalayıcı değildir; **klasik CV (Canny + kontur + homography + morfoloji)** ile **modern OCR**'ı tek bir doğru çalışan ürün altında birleştiren bir mühendislik gösterimidir. AirCanvas AI'ın "interaktif, jest tabanlı" tarafıyla **birbirini tamamlar**: AirCanvas oluşturma/çizim, SnapDoc tüketim/yapılandırma tarafıdır.
