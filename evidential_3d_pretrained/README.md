# Belirsizlik-Farkındalı 3B Nesne Tespiti — Pretrained Model

KITTI veri seti üzerinde Evidential Deep Learning tabanlı 3B nesne tespiti yapan
bir derin öğrenme projesidir. Bu klasör **Model 2**'nin (pretrained PointPillars
omurgası + sıfırdan eğitilen evidential başlık) kaynak kodunu içerir.

## Başlangıç

Eğer her şey kuruluysa, modelin canlı demosunu çalıştırmak için tek komut:

```bash
python tools/tester.py \
    --config configs/pretrained_kitti.yaml \
    --checkpoint output/pretrained_kitti_evidential/checkpoints/best_model.pth \
    --num_frames 20 --mode live
```

Bu komut KITTI val setinden 20 ardışık örneği sırayla modele verir, BEV (kuş
bakışı) tahminlerini bir matplotlib penceresinde 1.2 saniyede bir günceller.

---

## İçindekiler

1. [Sistem Gereksinimleri](#1-sistem-gereksinimleri)
2. [Kurulum](#2-kurulum)
3. [KITTI Veri Setinin İndirilmesi](#3-kitti-veri-setinin-indirilmesi)
4. [Pretrained Ağırlıkların İndirilmesi](#4-pretrained-ağırlıkların-indirilmesi)
5. [Eğitilmiş Modelin İndirilmesi](#5-eğitilmiş-modelin-indirilmesi)
6. [Çalıştırma — Hangi Komutla Ne Yapılır?](#6-çalıştırma--hangi-komutla-ne-yapılır)
7. [Klasör Yapısı](#7-klasör-yapısı)
8. [Sık Karşılaşılan Sorunlar](#8-sık-karşılaşılan-sorunlar)

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
ve **Google Colab Tesla T4** (eğitim).

---

## 2. Kurulum

### 2.1. Depoyu klonla

```bash
git clone <REPO_URL>
cd evidential_3d_pretrained
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
gdown>=4.7
imageio[ffmpeg]   # tester.py video modu için
```

> **Not:** `spconv` paketi gerekli **değildir** — bu proje saf PyTorch
> PointPillars implementasyonu kullanır.

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
evidential_3d_pretrained/data/kitti/
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

## 4. Pretrained Ağırlıkların İndirilmesi

OpenPCDet tarafından KITTI üzerinde 80 epoch eğitilmiş PointPillars
ağırlıklarına ihtiyaç vardır. İlk eğitimde otomatik indirilir; manuel indirme
isterseniz:

```bash
python -c "from models.pretrained_loader import download_pretrained; \
           download_pretrained('pretrained/pointpillar_7728.pth')"
```

Bu komut `gdown` üzerinden Google Drive'dan dosyayı (`~19 MB`) çeker. Drive
linki çalışmazsa otomatik olarak HuggingFace yedeğine düşer.

---

## 5. Eğitilmiş Modelin İndirilmesi

Demoyu çalıştırmak için kendi `best_model.pth` dosyanızı eğitebilir veya hazır
kontrol noktasını kullanabilirsiniz. Hazır kontrol noktası şu konumda olmalıdır:

```
output/pretrained_kitti_evidential/checkpoints/best_model.pth
```

Yoksa kendiniz eğitin:

```bash
python tools/train.py --config configs/pretrained_kitti.yaml
```

Yerel makinede (RTX 4050 6 GB) eğitim için config'de
`train.batch_size: 1` yapın, aksi takdirde VRAM dolar. Tam eğitim
yaklaşık 8 saat sürer (T4 üzerinde).

---

## 6. Çalıştırma — Hangi Komutla Ne Yapılır?

### 6.1. Canlı Demo (`tester.py`) — modelin çalışmasını izle

Bu, hocanıza/jüriye **modeli canlı göstermek** için tasarlanmıştır. KITTI val
örneklerini ardışık olarak modele verir, her birinin BEV tahminini ekrana basar.

#### Mod 1 — Live (matplotlib penceresi, otomatik geçiş)

```bash
python tools/tester.py \
    --config configs/pretrained_kitti.yaml \
    --checkpoint output/pretrained_kitti_evidential/checkpoints/best_model.pth \
    --num_frames 20 \
    --delay 1.5 \
    --mode live
```

Bir pencere açılır, 20 örnek 1.5'er saniye gösterilir. Tahmin kutuları
uncertainty'ye göre renklendirilir (yeşil = düşük belirsizlik, kırmızı = yüksek).
Mavi kesik çizgiler ground-truth kutuları temsil eder.

#### Mod 2 — Save (PNG olarak kaydet)

```bash
python tools/tester.py \
    --checkpoint output/pretrained_kitti_evidential/checkpoints/best_model.pth \
    --num_frames 30 \
    --mode save \
    --save_dir demo_frames
```

`demo_frames/` klasörüne 30 PNG yazar. Sunum slaytlarında kullanışlı.

#### Mod 3 — Video (MP4 oluştur)

```bash
python tools/tester.py \
    --checkpoint output/pretrained_kitti_evidential/checkpoints/best_model.pth \
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
| `--config` | `configs/pretrained_kitti.yaml` | Config yolu |
| `--checkpoint` | (zorunlu) | `.pth` dosyası |
| `--num_frames` | 20 | Gösterilecek örnek sayısı |
| `--start` | 0 | Val seti içindeki başlangıç indeksi |
| `--delay` | 1.2 | Live mod: çerçeveler arası saniye |
| `--mode` | `live` | `live` / `save` / `video` |
| `--save_dir` | `demo_frames` | save/video: PNG çıkış klasörü |
| `--output_path` | `demo.mp4` | video: çıktı dosyası |
| `--fps` | 2 | video: kare hızı |
| `--gpu` | 0 | CUDA cihaz indeksi |

### 6.2. Sıfırdan Eğitim

```bash
python tools/train.py --config configs/pretrained_kitti.yaml
```

Eğitim ilerlemesi `output/pretrained_kitti_evidential/logs/` klasörüne yazılır.
Auto-resume etkindir: eğitim kesilirse aynı komutu yeniden çalıştırmak son
checkpoint'ten devam ettirir.

### 6.3. Kapsamlı Değerlendirme (mAP, ECE, AUROC, AUSE)

```bash
python tools/evaluate.py \
    --config configs/pretrained_kitti.yaml \
    --checkpoint output/pretrained_kitti_evidential/checkpoints/best_model.pth \
    --visualize --num_viz 15
```

Tüm val seti üzerinde inference koşar, KITTI mAP hesaplar, belirsizlik
metriklerini (ECE, AUROC, AUSE) raporlar ve 15 BEV görselini diske kaydeder.

---

## 7. Klasör Yapısı

```
evidential_3d_pretrained/
├── configs/
│   └── pretrained_kitti.yaml       # Eğitim/değerlendirme config'i
├── data/
│   └── kitti/                      # Veri setini buraya yerleştirin (3. bölüm)
├── evaluation/
│   ├── calibration.py              # ECE, reliability diagram
│   └── uncertainty_metrics.py      # AUROC, sparsification
├── losses/
│   └── evidential_losses.py        # NIG + Dirichlet kayıpları
├── models/
│   ├── pretrained_backbone.py      # PillarVFE + Scatter + BEVBackbone
│   ├── pretrained_loader.py        # Pretrained ağırlık indirici/yükleyici
│   ├── evidential_head.py          # NIG + Dirichlet detection başlığı
│   ├── uncertainty_nms.py          # Belirsizlik-farkındalı NMS
│   └── uncertainty_detector.py     # Üst seviye detector sınıfı
├── notebooks/
│   └── pretrained_colab_setup.ipynb  # Colab tek-tıkla kurulum
├── pretrained/
│   └── pointpillar_7728.pth        # 4. bölümde indirilir
├── tools/
│   ├── kitti_dataset.py            # KITTI loader + voxelizasyon
│   ├── target_assigner.py          # CenterPoint hedef ataması
│   ├── train.py                    # Eğitim scripti
│   ├── evaluate.py                 # Değerlendirme scripti (mAP, UQ)
│   └── tester.py                   # Canlı demo scripti (BU DOSYA)
├── visualization/
│   └── vis_utils.py                # BEV görselleştirme
├── README.md                        # Bu dosya
└── requirements.txt
```

---

## 8. Sık Karşılaşılan Sorunlar

**Q: `ImportError: No module named 'models'`**
A: `tester.py`'i proje kök dizininden çalıştırın
(`evidential_3d_pretrained/` içinden), `tools/tester.py` olarak.

**Q: `RuntimeError: CUDA out of memory`**
A: Config'de `train.batch_size`'ı düşürün veya `eval.batch_size: 1` yapın.
Inference için bile 4 GB GPU yeterlidir.

**Q: matplotlib penceresi açılmıyor (`live` mod)**
A: Sunucu/SSH ortamındaysanız `--mode save` veya `--mode video` kullanın;
GUI olmadan PNG/MP4 çıktısı alırsınız.

**Q: `gdown: Cannot retrieve the public link`**
A: Google Drive bazen rate-limit uygular. `pretrained_loader.py` içindeki
HuggingFace yedek linki kullanılır, yoksa pretrained dosyasını manuel olarak
[OpenPCDet model zoo](https://github.com/open-mmlab/OpenPCDet)'undan indirip
`pretrained/pointpillar_7728.pth` konumuna yerleştirin.

**Q: KITTI klasör yapısı doğru ama "samples loaded: 0" diyor**
A: `ImageSets/val.txt` dosyasının var olduğundan ve içeriğinin (her satırda bir
sample ID) doğru olduğundan emin olun. Bu dosya repo içinde hazır gelir.

**Q: `imageio.imsave: ffmpeg not found`**
A: `pip install imageio[ffmpeg]` veya sisteme [ffmpeg](https://ffmpeg.org/)
kurun. Tester otomatik olarak GIF formatına geri düşer.

---

## Lisans ve Atıf

KITTI veri seti, [orijinal lisans şartları](https://www.cvlibs.net/datasets/kitti/)
altında dağıtılır. OpenPCDet pretrained ağırlıkları Apache 2.0 lisansı altındadır.
Bu projenin kaynak kodu eğitim/araştırma amaçlıdır.
