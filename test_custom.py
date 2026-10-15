"""
Kullanım:
  1. my_digits/ klasörüne el yazısı rakam görüntülerini koy.
     Dosya adı isteğe bağlıdır; örn: '3.png', 'digit1.jpg'
  2. Hangi modeli kullanmak istediğinizi belirtin (varsayılan: en iyi model)
     python3 test_custom.py --model D_Adam_Dropout

Görüntü gereksinimleri:
  - Siyah arka plan üzerinde beyaz rakam (MNIST formatı)
  - Veya beyaz arka plan üzerinde siyah rakam — script otomatik çevirir
"""

import os
import argparse
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from scipy import ndimage

from models import MLP, MODEL_CONFIGS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081


def load_model(config_name):
    config = MODEL_CONFIGS[config_name]
    model = MLP(
        hidden_layers=config["hidden_layers"],
        use_dropout=config["use_dropout"],
    ).to(DEVICE)
    weight_path = f"results/{config_name}.pt"
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Model ağırlıkları bulunamadı: {weight_path}\n"
            f"Önce 'python3 compare.py' çalıştırın."
        )
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()
    return model


def mnist_preprocess(path):
    """
    Görüntüyü gerçek MNIST formatına dönüştürür:
      1. Gri ton
      2. Arka plan beyazsa negatife çevir (siyah zemin, beyaz rakam)
      3. Gürültü temizleme (Gaussian blur + eşikleme)
      4. Sıkı bounding-box kırpması
      5. En-boy oranı korunarak 20×20 iç kutuya sığdır
      6. Ağırlık merkezine göre 28×28 çerçevede ortala
      7. MNIST normalizasyonu (μ=0.1307, σ=0.3081)
    """
    # 1. Gri ton
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)

    # 2. Arka plan tespiti — beyaz zemin ise negatife çevir
    if arr.mean() > 127:
        arr = 255.0 - arr

    # 3. Hafif bulanıklaştırma + Otsu benzeri eşikleme ile gürültü temizleme
    blurred = ndimage.gaussian_filter(arr, sigma=1)
    threshold = blurred.max() * 0.2
    binary = (blurred > threshold).astype(np.float32)

    # 4. Sıkı bounding-box: sıfır olmayan piksellerin sınırlarını bul
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any():
        # Tamamen boş görüntü → sıfır tensör döndür
        tensor = torch.zeros(1, 1, 28, 28, device=DEVICE)
        preview = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
        return tensor, preview

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    cropped = arr[r_min:r_max + 1, c_min:c_max + 1]

    # 5. En-boy oranını koruyarak 20×20 iç kutuya sığdır
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    digit_img = Image.fromarray(cropped.astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    )
    digit_arr = np.array(digit_img, dtype=np.float32)

    # 6. Ağırlık merkezini hesaplayıp 28×28 çerçevede ortala (MNIST standardı)
    canvas = np.zeros((28, 28), dtype=np.float32)
    cy_mass, cx_mass = ndimage.center_of_mass(digit_arr)
    # Merkez noktası 14,14 olacak şekilde yerleştir
    offset_r = int(round(14 - cy_mass))
    offset_c = int(round(14 - cx_mass))
    r0 = max(0, offset_r)
    c0 = max(0, offset_c)
    r1 = min(28, offset_r + new_h)
    c1 = min(28, offset_c + new_w)
    src_r0 = max(0, -offset_r)
    src_c0 = max(0, -offset_c)
    canvas[r0:r1, c0:c1] = digit_arr[src_r0:src_r0 + (r1 - r0), src_c0:src_c0 + (c1 - c0)]

    # 7. [0,255] → [0,1] → MNIST normalizasyonu
    canvas_norm = canvas / 255.0
    tensor = torch.tensor(canvas_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    tensor = tensor.to(DEVICE)

    # Görselleştirme için normalize edilmemiş 28×28 PIL görüntüsü
    preview = Image.fromarray(canvas.astype(np.uint8))
    return tensor, preview


def preprocess_image(path):
    return mnist_preprocess(path)


def predict(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
        pred = probs.argmax().item()
        conf = probs[pred].item()
    return pred, conf, probs.cpu().numpy()


def visualize_results(image_paths, previews, predictions, confidences, prob_lists, output_path):
    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
    if n == 1:
        axes = [axes]

    for i, (path, preview, pred, conf, probs) in enumerate(
        zip(image_paths, previews, predictions, confidences, prob_lists)
    ):
        # Sütun 0: Orijinal görüntü
        orig = Image.open(path).convert("L")
        axes[i][0].imshow(orig, cmap="gray")
        axes[i][0].set_title(f"Orijinal\n{os.path.basename(path)}", fontsize=10)
        axes[i][0].axis("off")

        # Sütun 1: MNIST formatına dönüştürülmüş 28×28 görüntü
        axes[i][1].imshow(np.array(preview), cmap="gray", vmin=0, vmax=255)
        axes[i][1].set_title(f"MNIST Formatı (28×28)\nTahmin: {pred}  ({conf*100:.1f}%)", fontsize=10)
        axes[i][1].axis("off")

        # Sütun 2: Sınıf olasılık çubuk grafiği
        colors = ["steelblue"] * 10
        colors[pred] = "tomato"
        axes[i][2].bar(range(10), probs * 100, color=colors)
        axes[i][2].set_xlabel("Rakam")
        axes[i][2].set_ylabel("Olasılık (%)")
        axes[i][2].set_xticks(range(10))
        axes[i][2].set_title("Sınıf Olasılıkları", fontsize=10)
        axes[i][2].grid(True, axis="y", alpha=0.3)

    plt.suptitle("El Yazısı Rakam Tanıma — Kendi Veriniz", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  → {output_path} kaydedildi")


def main():
    parser = argparse.ArgumentParser(description="El yazısı rakam testi")
    parser.add_argument("--model", default="D_Adam_Dropout",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Kullanılacak model (varsayılan: D_Adam_Dropout)")
    parser.add_argument("--dir", default="my_digits",
                        help="Görüntü dizini (varsayılan: my_digits/)")
    args = parser.parse_args()

    image_paths = sorted(
        glob.glob(os.path.join(args.dir, "*.png")) +
        glob.glob(os.path.join(args.dir, "*.jpg")) +
        glob.glob(os.path.join(args.dir, "*.jpeg"))
    )

    if not image_paths:
        print(f"\n  '{args.dir}/' klasöründe PNG/JPG dosyası bulunamadı.")
        print("  El yazısı rakam görüntülerinizi o klasöre koyun ve tekrar çalıştırın.\n")
        return

    print(f"\nModel yükleniyor: {args.model}")
    model = load_model(args.model)

    predictions, confidences, prob_lists, previews = [], [], [], []
    print(f"\n{'Dosya':<30} {'Tahmin':>7} {'Güven':>8}")
    print("-" * 48)
    for path in image_paths:
        tensor, preview = preprocess_image(path)
        pred, conf, probs = predict(model, tensor)
        predictions.append(pred)
        confidences.append(conf)
        prob_lists.append(probs)
        previews.append(preview)
        print(f"{os.path.basename(path):<30} {pred:>7}   {conf*100:>6.1f}%")

    output_path = "results/custom_test_results.png"
    visualize_results(image_paths, previews, predictions, confidences, prob_lists, output_path)
    print(f"\nToplam {len(image_paths)} görüntü işlendi.")
    print("Not: Eğitim verisi (MNIST) ile kendi el yazınız arasındaki")
    print("     fark (distribution shift) düşük güven skorlarında görülebilir.\n")


if __name__ == "__main__":
    main()
