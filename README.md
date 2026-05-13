# Belirsizlik-Farkındalı 3B Nesne Tespiti

KITTI LiDAR veri seti üzerinde **Evidential Deep Learning (EDL)** tabanlı, belirsizlik kestirimi
yapan 3B nesne tespit projesi. Repo iki farklı modeli barındırır ve her iki model de aynı
LiDAR girdisini farklı omurga–başlık kombinasyonlarıyla işleyerek hem `Car / Pedestrian / Cyclist`
sınıflarını tespit eder hem de her tespit için **epistemic + aleatoric belirsizlik** üretir.

> Bu README, GitHub ana sayfasında görünen tek dosya olduğu için repo kullanımının **tamamını**
> kapsayacak şekilde hazırlanmıştır. Alt klasörlerdeki `evidential_3d_pretrained/README.md` ve
> `uncertainty_3d_detection/README.md` dosyaları her modelin kendi ayrıntılarını içerir.

---

## İçindekiler

1. [Hızlı Başlangıç](#1-hızlı-başlangıç)
2. [Projedeki İki Model](#2-projedeki-iki-model)
3. [Sistem Gereksinimleri](#3-sistem-gereksinimleri)
4. [Kurulum (Adım Adım)](#4-kurulum-adım-adım)
5. [KITTI Veri Setinin İndirilmesi ve Yerleştirilmesi](#5-kitti-veri-setinin-indirilmesi-ve-yerleştirilmesi)
6. [Eğitilmiş Modeller](#6-eğitilmiş-modeller)
7. [Çalıştırma — Komutlar ve Modlar](#7-çalıştırma--komutlar-ve-modlar)
8. [Argüman Referansı (tester.py)](#8-argüman-referansı-testerpy)
9. [Çıktıların Yorumlanması](#9-çıktıların-yorumlanması)
10. [Klasör Yapısı](#10-klasör-yapısı)
11. [Kısıtlamalar ve Uyarılar](#11-kısıtlamalar-ve-uyarılar)
12. [Lisans ve Atıf](#12-lisans-ve-atıf)

---

## 1. Hızlı Başlangıç

Bağımlılıklar kuruluysa ve KITTI verisi yerindeyse, demo tek komutla çalışır:

**Model 1 — Sıfırdan eğitilen CenterPoint + EDL**

```bash
cd uncertainty_3d_detection
python tester.py --mode live --num_frames 20
```

**Model 2 — Pretrained PointPillars + EDL başlık**

```bash
cd evidential_3d_pretrained
python tester.py --mode live --num_frames 20
```

Her iki komut da KITTI val setinden ardışık 20 örneği modele verir ve kuş bakışı (BEV)
tahminleri 1.2 saniyede bir matplotlib penceresinde günceller.

---

## 2. Projedeki İki Model

Aynı görev (KITTI 3B tespiti + belirsizlik kestirimi) iki farklı yaklaşımla çözülmüştür.
İkisinin de kaynak kodu bu repoda bulunur.

### Model 1 — `uncertainty_3d_detection/`
- **Omurga:** Sıfırdan eğitilmiş 2-ölçekli PointPillars (parametre sayısı ≈ 1.2M)
- **Başlık:** Anchor-free CenterPoint tarzı + Normal-Inverse-Gamma (NIG) regresyonu
  ve Dirichlet sınıflandırma evidential başlık
- **Eğitim:** Uçtan uca, 28 epoch
- **Güçlü yanı:** **Daha iyi belirsizlik kalibrasyonu** (AUROC ≈ 0.687, AUSE ≈ 0.194)
- **Zayıf yanı:** Düşük mAP (≈ %37)

### Model 2 — `evidential_3d_pretrained/`
- **Omurga:** OpenPCDet `pointpillar_7728.pth` (KITTI üzerinde 80 epoch eğitilmiş)
- **Başlık:** Aynı NIG + Dirichlet evidential başlık, sıfırdan eğitildi
- **Eğitim:** İki aşamalı (5 epoch frozen + 20 epoch fine-tune) veya 20 epoch
  yalnızca frozen
- **Güçlü yanı:** **Daha yüksek mAP** (≈ %55)
- **Zayıf yanı:** Belirsizlik sinyali zayıflar (AUROC ≈ 0.50–0.53)

| | Model 1 | Model 2 |
|---|---|---|
| Klasör | `uncertainty_3d_detection/` | `evidential_3d_pretrained/` |
| Omurga | Sıfırdan | Pretrained (OpenPCDet) |
| mAP | %37.35 | %54.69 |
| AUROC | **0.687** | 0.498–0.528 |
| ECE | 0.116 | 0.127 |
| Tipik kullanım | Belirsizlik analizi | Tespit doğruluğu |

---

## 3. Sistem Gereksinimleri

| Bileşen | Minimum | Önerilen |
|---|---|---|
| Python | 3.9 | 3.10 / 3.11 |
| CUDA | — (CPU çalışır ama yavaş) | CUDA 11.8+ |
| GPU VRAM | 4 GB (sadece inference) | 8 GB+ (eğitim için) |
| RAM | 8 GB | 16 GB+ |
| Disk | 30 GB (KITTI dahil) | 50 GB |
| İşletim Sistemi | Linux / Windows / WSL2 | Linux veya WSL2 |

Geliştirme/test ortamı: **NVIDIA GeForce RTX 4050 Laptop GPU** (yerel inference) ve
**Google Colab Tesla T4 / A100** (eğitim).

---

## 4. Kurulum (Adım Adım)

### 4.1. Depoyu Klonla

```bash
git clone <REPO_URL>
cd <REPO_KLASORU>
```

Bu noktada `evidential_3d_pretrained/`, `uncertainty_3d_detection/` ve `figures/` klasörlerini
ve bu README'yi görmelisin.

### 4.2. Sanal Ortam Oluştur

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 4.3. PyTorch'u CUDA ile Kur

PyTorch'u CUDA sürümünüze uygun olarak [pytorch.org](https://pytorch.org) üzerinden kurun.
CUDA 11.8 için örnek:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CPU-only istiyorsanız:

```bash
pip install torch torchvision
```

### 4.4. Kalan Bağımlılıkları Yükle

İki modelin gereksinimleri büyük ölçüde aynıdır; her ikisi için de aynı sanal ortamda kurabilirsiniz:

```bash
pip install -r evidential_3d_pretrained/requirements.txt
pip install -r uncertainty_3d_detection/requirements.txt
```

Ortak bağımlılıklar:
```
torch>=1.10
torchvision>=0.11
numpy>=1.21
scipy>=1.7
scikit-learn>=1.0
matplotlib>=3.5
opencv-python>=4.5
open3d>=0.15
pyyaml>=5.4
tqdm>=4.62
tensorboardX>=2.5
easydict>=1.9
numba>=0.53
gdown>=4.7         # Yalnızca Model 2: pretrained ağırlık indirme
imageio[ffmpeg]    # tester.py video modu için
```

> **Not:** `spconv` paketi gerekli **değildir** — her iki proje de saf PyTorch tabanlı
> PointPillars/CenterPoint implementasyonu kullanır.

---

## 5. KITTI Veri Seti (Sadece Yeniden Eğitim veya Geniş Test İçin)

> **Demo için gerek yok.** Repo, val setinin ilk 20 örneğini (velodyne + label_2 + calib) hazır
> içerir; `tester.py` doğrudan bu örnekler üzerinde çalışır. Aşağıdaki indirme adımları
> **yalnızca** şu durumlar için gereklidir:
> - Demo'yu 20 örnekten daha geniş bir alt küme üzerinde denemek
> - Tüm val seti üzerinde değerlendirme yapmak (`tools/evaluate.py`)
> - Modelleri **sıfırdan yeniden eğitmek** (`tools/train.py`)

Bu durumlardan biri geçerliyse KITTI 3D Object Detection benchmark'ını resmi siteden indirin:
<https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>
(ücretsiz, e-posta ile kayıt gerekiyor).

İndirilmesi gereken dört zip:

| Dosya | İçerik | Boyut |
|---|---|---|
| `data_object_velodyne.zip` | LiDAR nokta bulutları | ~12 GB |
| `data_object_label_2.zip` | Etiketler (txt) | ~5 MB |
| `data_object_calib.zip` | Kamera kalibrasyonu | ~16 MB |
| `data_object_image_2.zip` | RGB görüntüler (opsiyonel) | ~12 GB |

İçerikleri **her iki proje klasörünün altında** aşağıdaki yapıya çıkartın:

```
<model_klasörü>/data/kitti/
├── ImageSets/
│   ├── train.txt        # repo ile birlikte gelir
│   └── val.txt          # repo ile birlikte gelir
├── training/
│   ├── velodyne/        # 0000000.bin, 0000001.bin, ...
│   ├── label_2/         # 0000000.txt, ...
│   ├── calib/           # 0000000.txt, ...
│   └── image_2/         # opsiyonel
└── testing/             # opsiyonel
```

`ImageSets/train.txt` ve `val.txt` dosyaları KITTI'nin standart 3712/3769 ayrımına karşılık
gelir (Chen et al., 2017) ve repo içinde hazır gelir.

> **Not:** Aynı veri seti **iki defa** yerleştirilmelidir — `evidential_3d_pretrained/data/kitti/`
> ve `uncertainty_3d_detection/data/kitti/` altına. Disk yerinden tasarruf etmek için sembolik
> link kullanabilirsiniz (Linux/WSL: `ln -s`; Windows: `mklink /D`). İndirilen dosyalar mevcut
> 20 örneğin üzerine yazılır, sorun değildir.

---

## 6. Eğitilmiş Modeller

Repo, her iki model için de eğitilmiş `best_model.pth` kontrol noktası içerir. Bu dosyaların
**aşağıdaki tam konumlarda** olması zorunludur:

| Model | Checkpoint Yolu |
|---|---|
| Model 1 | `uncertainty_3d_detection/Model/best_model.pth` |
| Model 2 | `evidential_3d_pretrained/Model/best_model.pth` |

> **Not:** Eğitim sırasında üretilen ara checkpoint'ler (`output/.../checkpoints/`) dahil
> değildir; sadece `Model/best_model.pth` dosyası demoyu çalıştırmak için yeterlidir.

---

## 7. Çalıştırma — Komutlar ve Modlar

Her iki proje klasöründe **aynı isimde** ve **aynı argüman setine sahip** bir `tester.py`
dosyası vardır. Komutlar ilgili proje klasörüne `cd` ile girdikten sonra çalıştırılır.

### 7.1. Live Mod (matplotlib penceresi)

Tahminlerin canlı izlendiği moddur. Her örnek `--delay` saniye boyunca ekranda kalır,
ardından bir sonraki örneğe geçilir.

```bash
# Model 1
cd uncertainty_3d_detection
python tester.py --mode live --num_frames 20 --delay 1.5

# Model 2
cd evidential_3d_pretrained
python tester.py --mode live --num_frames 20 --delay 1.5
```

> **Gerekli koşul:** GUI destekli bir ortam (yerel makine veya X-forwarding yapılan SSH).
> Sunucuda doğrudan çalıştırıyorsanız Save veya Video modunu kullanın.

### 7.2. Save Mod (PNG çıktı)

Her örneği `--save_dir` klasörüne PNG olarak yazar.

```bash
python tester.py --mode save --num_frames 30 --save_dir demo_frames
```

### 7.3. Video Mod (MP4 oluştur)

Tüm çerçeveleri yazıp ardından `--output_path` adlı videoyu üretir. ffmpeg yoksa otomatik
olarak `.gif` çıktısına geri düşer.

```bash
python tester.py --mode video --num_frames 50 --output_path demo.mp4 --fps 2
```

### 7.4. Sıfırdan Eğitim (opsiyonel)

Repodaki `best_model.pth` zaten eğitilmiş halde gelir. Sıfırdan eğitmek isterseniz:

```bash
# Model 1
cd uncertainty_3d_detection
python tools/train.py --config configs/centerpoint_kitti.yaml

# Model 2
cd evidential_3d_pretrained
python tools/train.py --config configs/pretrained_kitti.yaml
```

Auto-resume mekanizması her iki projede etkindir: aynı komutu yeniden çalıştırmak son
checkpoint'ten devam eder.

### 7.5. Kapsamlı Değerlendirme (mAP, ECE, AUROC, AUSE)

```bash
# Model 1
python tools/evaluate.py --config configs/centerpoint_kitti.yaml \
    --checkpoint Model/best_model.pth --visualize --num_viz 15

# Model 2
python tools/evaluate.py --config configs/pretrained_kitti.yaml \
    --checkpoint Model/best_model.pth --visualize --num_viz 15
```

---

## 8. Argüman Referansı (tester.py)

Her iki projedeki `tester.py` aynı argüman setini kullanır:

| Argüman | Tip | Varsayılan | Açıklama |
|---|---|---|---|
| `--config` | str | `configs/<model>_kitti.yaml` | YAML config yolu |
| `--checkpoint` | str | `Model/best_model.pth` | `.pth` dosyası |
| `--num_frames` | int | 20 | Gösterilecek örnek sayısı |
| `--start` | int | 0 | Val seti içindeki başlangıç indeksi |
| `--delay` | float | 1.2 | Live mod: çerçeveler arası saniye |
| `--mode` | str | `live` | `live` / `save` / `video` |
| `--save_dir` | str | `demo_frames` | Save/video modu: PNG klasörü |
| `--output_path` | str | `demo.mp4` | Video modu: çıktı dosyası |
| `--fps` | int | 2 | Video modu: kare hızı |
| `--gpu` | int | 0 | CUDA cihaz indeksi |

Tüm yol argümanları **script'in bulunduğu klasöre göre** çözümlenir; yani komutu hangi
working directory'den çalıştırırsanız çalıştırın aynı sonucu alırsınız.

---

## 9. Çıktıların Yorumlanması

`tester.py` çıktısındaki BEV görselleştirmesi şunları içerir:

- **Gri/siyah nokta bulutu:** KITTI Velodyne'den gelen ham LiDAR noktaları (kuş bakışı projeksiyon).
- **Renkli kutular (tahminler):** Modelin ürettiği 3B sınırlayıcı kutuların BEV izdüşümü.
  Renk, **her tahminin belirsizliğini** kodlar — yeşil = düşük belirsizlik, sarı = orta,
  kırmızı = yüksek.
- **Mavi kesik çizgili kutular:** Ground-truth (gerçek) etiketler. Karşılaştırma için
  birlikte çizilir.
- **Sağ taraftaki colorbar:** Belirsizlik renk skalasının sayısal karşılığı (0 = emin,
  yüksek değer = belirsiz).
- **Başlık:** `KITTI val/<örnek_id> | <tahmin_sayısı> tahmin`
- **Terminal log'u:** `inf` = inference süresi (ms), `unc=[min, max]` = o örnekteki tahminlerin
  belirsizlik aralığı.

---

## 10. Klasör Yapısı

```
<REPO_KÖKÜ>/
├── README.md                                  # Bu dosya
├── figures/                                   
│
├── evidential_3d_pretrained/                  # MODEL 2
│   ├── configs/pretrained_kitti.yaml
│   ├── data/kitti/...                         # KITTI buraya (Bölüm 5)
│   ├── Model/best_model.pth                   # Eğitilmiş model (Bölüm 6)
│   ├── pretrained/pointpillar_7728.pth        # OpenPCDet ağırlığı (eğitim için)
│   ├── models/
│   │   ├── pretrained_backbone.py             # PillarVFE + Scatter + BEVBackbone
│   │   ├── pretrained_loader.py               # OpenPCDet ağırlığı yükleyici
│   │   ├── evidential_head.py                 # NIG + Dirichlet başlık
│   │   ├── uncertainty_nms.py                 # Belirsizlik-farkındalı NMS
│   │   └── uncertainty_detector.py            # Üst seviye detector
│   ├── losses/evidential_losses.py
│   ├── evaluation/{calibration.py, uncertainty_metrics.py}
│   ├── visualization/vis_utils.py
│   ├── tools/{kitti_dataset.py, target_assigner.py, train.py, evaluate.py}
│   ├── tester.py
│   ├── requirements.txt
│   └── README.md
│
└── uncertainty_3d_detection/                  # MODEL 1
    ├── configs/centerpoint_kitti.yaml
    ├── data/kitti/...                         # KITTI buraya (Bölüm 5)
    ├── Model/best_model.pth                   # Eğitilmiş model (Bölüm 6)
    ├── models/
    │   ├── uncertainty_centerpoint.py         # Ana detector
    │   ├── evidential_head.py
    │   ├── mc_dropout.py                      # MC Dropout wrapper (ablasyon)
    │   └── uncertainty_nms.py
    ├── losses/evidential_losses.py
    ├── evaluation/{calibration.py, uncertainty_metrics.py}
    ├── visualization/vis_utils.py
    ├── tools/{kitti_dataset.py, target_assigner.py, train.py, evaluate.py}
    ├── tester.py
    ├── requirements.txt
    └── README.md
```

---

## 11. Kısıtlamalar ve Uyarılar

Aşağıdaki maddeler reponun **bilinen sınırlarını** belirler. Demo çalıştırırken veya
sonuçları yorumlarken bu kısıtlamaları göz önünde bulundurun.

### 11.1. Donanım ve Bellek

- **VRAM 4 GB'tan azsa** Model 2 inference sırasında bile bellek hatası verebilir.
  Bu durumda config dosyasında `eval.batch_size: 1` yapın.
- **VRAM 6 GB civarındaysa** (örn. RTX 4050) eğitim için `train.batch_size: 1` zorunludur.
  Daha yüksek batch_size CUDA out-of-memory ile sonuçlanır.
- Yeterli VRAM olmadığı durumda CPU üzerinde de çalışır; ancak bir BEV karesi 30 saniyeyi
  bulabilir.

### 11.2. Veri Seti

- Her iki model yalnızca **KITTI** veri seti üzerinde eğitilmiş ve doğrulanmıştır. nuScenes,
  Waymo Open Dataset veya Argoverse gibi başka veri setleriyle çalışmaz; bu setlerin nokta
  bulutu formatı, sensör konumu ve sınıf seti farklıdır.
- Sınıflar yalnızca `Car`, `Pedestrian`, `Cyclist` ile sınırlıdır. KITTI'deki diğer sınıflar
  (`Van`, `Truck`, `Tram`, `Person_sitting`, `Misc`) eğitim sırasında filtrelenir.
- Çalışma alanı yalnızca aracın **ön cephesidir** (180°): `x ∈ [0, 69.12]`,
  `y ∈ [-39.68, 39.68]`, `z ∈ [-3, 1]` metre. Bu aralığın dışındaki noktalar otomatik atılır.
- KITTI val seti **3769 örnek**, train seti **3712 örnek** içerir (standart Chen et al. 2017
  ayrımı).

### 11.3. Modellerin Performans Sınırları

- Model 1: mAP ≈ %37.35 — düşük; üretim/otonom sürüş için yeterli değildir, akademik
  ablasyon için tasarlanmıştır.
- Model 2: mAP ≈ %54.69 — pretrained PointPillars'ın orijinal %77 mAP'inin altında. Sebep:
  evidential başlık sıfırdan eğitildiği için pretrained başlığın katkısı kaybedilir.
- Hiçbir modelde AUROC > 0.7 olarak ölçülmemiştir; ideal değer 1.0'dır.
- Model 2 v2'de AUROC < 0.5 — yani belirsizlik sinyali kısmen **terstir** (TP belirsizliği
  FP belirsizliğinden yüksek). Bu, raporda *evidence collapse* olarak tartışılır.

### 11.4. Çalışma Ortamı

- `tester.py` mutlaka **ilgili proje klasörünün içinden** çalıştırılmalıdır:
  ```bash
  cd evidential_3d_pretrained
  python tester.py ...        # ✅ doğru
  
  python evidential_3d_pretrained/tester.py ...   # ⚠️ yine çalışır ama log mesajları farklı görünebilir
  ```
- Live mod için **GUI destekli ortam zorunludur**. Headless sunucularda matplotlib
  penceresi açılmaz; `--mode save` veya `--mode video` kullanın.
- Video modu ffmpeg gerektirir. Yüklü değilse otomatik olarak `.gif` çıktısına geri düşer.

### 11.5. Eğitim Süresi

- Model 1 (sıfırdan, 28 epoch): Tesla T4 üzerinde ≈ **10–12 saat**.
- Model 2 (5 frozen + 20 fine-tune): Tesla T4 üzerinde ≈ **8 saat**.
- Auto-resume etkin olsa da Colab oturum süresi sınırlıdır; tek seansta tam eğitim
  garantili değildir.

### 11.6. Bağımlılık Tuzakları

- PyTorch sürümü < 1.10 desteklenmez. `torch.amp` ve modern `torch.nn.functional` API'leri
  kullanılır.
- numpy 2.x ile bazı uyumsuzluklar görülebilir; sorun olursa `numpy<2.0` sürümüne sabitleyin.
- `open3d` opsiyoneldir; yalnızca PCD görselleştirme ek modülleri için gerekir. Yüklenmesi
  çekirdek demo'yu etkilemez.

### 11.7. Yapılmaması Gerekenler

- **Model 1 checkpoint'ini Model 2 ile yüklemeyin** (veya tersi). Mimari farklıdır;
  `state_dict` uyuşmaz ve sessiz şekilde yanlış sonuçlar üretir.
- **Augmentation parametrelerini ekstrem değiştirmeyin.** Özellikle Model 2'de
  `gt-sampling` kapatılırsa Cyclist AP sıfıra düşer.
- **`λ_KL` parametresini sıfırlamayın.** KL düzenleyici olmadan evidential sınıflandırma
  hızla collapse olur; AUROC 0.5'in altına iner.
- **Aşama 2'yi (fine-tuning) tek aşamada eğitime dahil etmeyin.** Önce 5 epoch frozen
  ısınma yapılmazsa pretrained omurganın özellikleri bozulur.

---

## 12. Lisans ve Atıf

- KITTI veri seti [orijinal lisans şartları](https://www.cvlibs.net/datasets/kitti/)
  altında dağıtılır.
- OpenPCDet pretrained PointPillars ağırlıkları Apache 2.0 lisansı altındadır.
- Bu projenin kaynak kodu eğitim/araştırma amaçlıdır.
