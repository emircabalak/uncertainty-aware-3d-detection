# Belirsizlik-Farkındalı 3B Nesne Tespiti — CenterPoint (Model 1)

KITTI veri seti üzerinde Evidential Deep Learning tabanlı 3B nesne tespiti yapan
bir derin öğrenme projesidir. Bu klasör **Model 1**'in (sıfırdan eğitilen
CenterPoint omurgası + evidential başlık) kaynak kodunu içerir.

## Başlangıç

Eğer her şey kuruluysa, modelin canlı demosunu çalıştırmak için tek komut:

```bash
python tester.py \
    --config configs/centerpoint_kitti.yaml \
    --checkpoint Model/best_model.pth \
    --num_frames 20 --mode live
```

Bu komut KITTI val setinden 20 ardışık örneği sırayla modele verir, BEV (kuş
bakışı) tahminlerini bir matplotlib penceresinde 1.2 saniyede bir günceller.

---

## İçindekiler

1. [Sistem Gereksinimleri](#1-sistem-gereksinimleri)
2. [Kurulum](#2-kurulum)
3. [KITTI Veri Setinin İndirilmesi](#3-kitti-veri-setinin-indirilmesi)
4. [Eğitilmiş Modelin Yerleştirilmesi](#4-eğitilmiş-modelin-yerleştirilmesi)
5. [Çalıştırma — Hangi Komutla Ne Yapılır?](#5-çalıştırma--hangi-komutla-ne-yapılır)
6. [Klasör Yapısı](#6-klasör-yapısı)
7. [Sık Karşılaşılan Sorunlar](#7-sık-karşılaşılan-sorunlar)

---

## 1. Sistem Gereksinimleri

| Bileşen | Minimum | Önerilen |
|---|---|---|
| Python | 3.9 | 3.10 / 3.11 |
| CUDA | — (CPU çalışır ama yavaş) | CUDA 11.8+ |
| GPU VRAM | 4 GB (sadece inference) | 8 GB+ (eğitim için) |
| RAM | 8 GB | 16 GB+ |
| Disk | 30 GB (KITTI dahil) | 50 GB |
| İşletim Sistemi | Linux / Windows / WSL2 | Linux veya WSL2 |

Geliştirme ve test ortamı: **NVIDIA GeForce RTX 4050 Laptop GPU** (yerel inference)
ve **Google Colab Tesla T4 / A100** (eğitim).

> **Model 1 vs Model 2:** Bu model omurgasını da sıfırdan eğitir (pretrained
> ağırlık kullanmaz). Daha iyi belirsizlik kalibrasyonu sağlar; detection
> metriği (mAP) bakımından Model 2'den daha düşük olabilir.

---

## 2. Kurulum

### 2.1. Depoyu klonla

```bash
git clone <REPO_URL>
cd uncertainty_3d_detection
```

### 2.2. Sanal ortam oluştur (önerilen)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2.3. PyTorch'u CUDA ile kur

PyTorch'u kendi CUDA sürümünüze uygun olarak [pytorch.org](https://pytorch.org)
üzerinden kurun. CUDA 11.8 için örnek:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CPU-only istiyorsanız:

```bash
pip install torch torchvision
```

### 2.4. Kalan bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

`requirements.txt` içeriği:

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
imageio[ffmpeg]   # tester.py video modu için
```

> **Not:** `spconv` paketi gerekli **değildir** — bu proje saf PyTorch
> PointPillars/CenterPoint implementasyonu kullanır.

---

## 3. KITTI Veri Setinin İndirilmesi

Resmi siteden indirme: <https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>
(ücretsiz, e-posta ile kayıt gerekiyor.)

İndirilmesi gereken dört dosya:

| Dosya | Boyut |
|---|---|
| `data_object_velodyne.zip` (LiDAR nokta bulutları) | ~12 GB |
| `data_object_label_2.zip` (etiketler) | ~5 MB |
| `data_object_calib.zip` (kamera kalibrasyonu) | ~16 MB |
| `data_object_image_2.zip` (RGB görüntüler — opsiyonel) | ~12 GB |

Aşağıdaki yapıya çıkartın:

```
uncertainty_3d_detection/data/kitti/
├── ImageSets/
│   ├── train.txt        # bu repo içinde sağlanmıştır
│   └── val.txt
├── training/
│   ├── velodyne/        # 0000000.bin, 0000001.bin, ...
│   ├── label_2/         # 0000000.txt, ...
│   ├── calib/           # 0000000.txt, ...
│   └── image_2/         # opsiyonel
└── testing/             # opsiyonel
```

`ImageSets/train.txt` ve `val.txt` dosyaları, KITTI'nin standart 3712/3769
ayrımına karşılık gelir (Chen et al., 2017). Bu dosyalar repo içinde hazır
gelir.

> **Pratik tüyo:** Eğer sadece tester.py'i denemek istiyorsanız, tüm KITTI
> yerine yalnızca `velodyne/` + `label_2/` + `calib/` klasörlerini indirmeniz
> yeterlidir.

---

## 4. Eğitilmiş Modelin Yerleştirilmesi

Demo çalıştırmak için eğitilmiş kontrol noktası şu konumda olmalıdır:

```
uncertainty_3d_detection/Model/best_model.pth
```

`Model/` klasörü yoksa oluşturun ve `best_model.pth` dosyasını içine koyun.
Kendiniz eğitmek isterseniz:

```bash
python tools/train.py --config configs/centerpoint_kitti.yaml
```

Yerel makinede (RTX 4050 6 GB) eğitim için config'de
`train.batch_size: 1` yapın, aksi takdirde VRAM dolar. Tam eğitim
yaklaşık 10–12 saat sürer (T4 üzerinde 30 epoch).

---

## 5. Çalıştırma — Hangi Komutla Ne Yapılır?

### 5.1. Canlı Demo (`tester.py`) — modelin çalışmasını izle

Bu, hocanıza/jüriye **modeli canlı göstermek** için tasarlanmıştır. KITTI val
örneklerini ardışık olarak modele verir, her birinin BEV tahminini ekrana basar.

#### Mod 1 — Live (matplotlib penceresi, otomatik geçiş)

```bash
python tester.py \
    --config configs/centerpoint_kitti.yaml \
    --checkpoint Model/best_model.pth \
    --num_frames 20 \
    --delay 1.5 \
    --mode live
```

Bir pencere açılır, 20 örnek 1.5'er saniye gösterilir. Tahmin kutuları
uncertainty'ye göre renklendirilir (yeşil = düşük belirsizlik, kırmızı = yüksek).
Mavi kesik çizgiler ground-truth kutuları temsil eder.

#### Mod 2 — Save (PNG olarak kaydet)

```bash
python tester.py \
    --checkpoint Model/best_model.pth \
    --num_frames 30 \
    --mode save \
    --save_dir demo_frames
```

`demo_frames/` klasörüne 30 PNG yazar. Sunum slaytlarında kullanışlı.

#### Mod 3 — Video (MP4 oluştur)

```bash
python tester.py \
    --checkpoint Model/best_model.pth \
    --num_frames 50 \
    --mode video \
    --output_path demo.mp4 \
    --fps 2
```

Tüm çerçeveleri kaydeder ve `demo.mp4` oluşturur (ffmpeg gerektirir;
yüklü değilse otomatik olarak `demo.gif` üretir).

#### Tüm tester.py argümanları

| Argüman | Varsayılan | Açıklama |
|---|---|---|
| `--config` | `configs/centerpoint_kitti.yaml` | Config yolu |
| `--checkpoint` | `Model/best_model.pth` | `.pth` dosyası |
| `--num_frames` | 20 | Gösterilecek örnek sayısı |
| `--start` | 0 | Val seti içindeki başlangıç indeksi |
| `--delay` | 1.2 | Live mod: çerçeveler arası saniye |
| `--mode` | `live` | `live` / `save` / `video` |
| `--save_dir` | `demo_frames` | save/video: PNG çıkış klasörü |
| `--output_path` | `demo.mp4` | video: çıktı dosyası |
| `--fps` | 2 | video: kare hızı |
| `--gpu` | 0 | CUDA cihaz indeksi |

### 5.2. Sıfırdan Eğitim

```bash
python tools/train.py --config configs/centerpoint_kitti.yaml
```

Eğitim ilerlemesi `output/centerpoint_kitti_evidential/logs/` klasörüne yazılır.
Auto-resume etkindir: eğitim kesilirse aynı komutu yeniden çalıştırmak son
checkpoint'ten devam ettirir.

### 5.3. Kapsamlı Değerlendirme (mAP, ECE, AUROC, AUSE)

```bash
python tools/evaluate.py \
    --config configs/centerpoint_kitti.yaml \
    --checkpoint Model/best_model.pth \
    --visualize --num_viz 15
```

Tüm val seti üzerinde inference koşar, KITTI mAP hesaplar, belirsizlik
metriklerini (ECE, AUROC, AUSE) raporlar ve 15 BEV görselini diske kaydeder.

---

## 6. Klasör Yapısı

```
uncertainty_3d_detection/
├── configs/
│   └── centerpoint_kitti.yaml      # Eğitim/değerlendirme config'i
├── data/
│   └── kitti/                      # Veri setini buraya yerleştirin (3. bölüm)
├── evaluation/
│   ├── calibration.py              # ECE, reliability diagram
│   └── uncertainty_metrics.py      # AUROC, sparsification
├── losses/
│   └── evidential_losses.py        # NIG + Dirichlet kayıpları
├── models/
│   ├── uncertainty_centerpoint.py  # Ana detector sınıfı (UncertaintyCenterPoint)
│   ├── evidential_head.py          # NIG + Dirichlet detection başlığı
│   ├── mc_dropout.py               # MC Dropout wrapper
│   └── uncertainty_nms.py          # Belirsizlik-farkındalı NMS
├── Model/
│   └── best_model.pth              # 4. bölümde yerleştirilir
├── notebooks/
│   ├── 00_download_kitti.ipynb     # KITTI indirme yardımcısı
│   └── 01_colab_setup.ipynb        # Colab tek-tıkla kurulum
├── tools/
│   ├── kitti_dataset.py            # KITTI loader + voxelizasyon
│   ├── target_assigner.py          # CenterPoint hedef ataması
│   ├── train.py                    # Eğitim scripti
│   └── evaluate.py                 # Değerlendirme scripti (mAP, UQ)
├── visualization/
│   └── vis_utils.py                # BEV görselleştirme
├── tester.py                       # Canlı demo scripti (BU DOSYA)
├── README.md                       # Bu dosya
└── requirements.txt
```

---

## 7. Sık Karşılaşılan Sorunlar

**Q: `ImportError: No module named 'models'`**
A: `tester.py`'i proje kök dizininden çalıştırın
(`uncertainty_3d_detection/` içinden), `python tester.py` olarak.

**Q: `RuntimeError: CUDA out of memory`**
A: Config'de `train.batch_size`'ı düşürün veya `eval.batch_size: 1` yapın.
Inference için bile 4 GB GPU yeterlidir.

**Q: matplotlib penceresi açılmıyor (`live` mod)**
A: Sunucu/SSH ortamındaysanız `--mode save` veya `--mode video` kullanın;
GUI olmadan PNG/MP4 çıktısı alırsınız.

**Q: KITTI klasör yapısı doğru ama "samples loaded: 0" diyor**
A: `ImageSets/val.txt` dosyasının var olduğundan ve içeriğinin (her satırda bir
sample ID) doğru olduğundan emin olun. Bu dosya repo içinde hazır gelir.

**Q: `imageio.imsave: ffmpeg not found`**
A: `pip install imageio[ffmpeg]` veya sisteme [ffmpeg](https://ffmpeg.org/)
kurun. Tester otomatik olarak GIF formatına geri düşer.

**Q: `KeyError: 'final_boxes'` veya benzer bir anahtar hatası**
A: Checkpoint'in bu repo ile eğitilmiş olduğundan emin olun. Model 2'nin
(`evidential_3d_pretrained`) checkpoint'i Model 1 ile uyumlu değildir.

---

## Lisans ve Atıf

KITTI veri seti, [orijinal lisans şartları](https://www.cvlibs.net/datasets/kitti/)
altında dağıtılır. Bu projenin kaynak kodu eğitim/araştırma amaçlıdır.
