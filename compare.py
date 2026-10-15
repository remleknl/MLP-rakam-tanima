import os
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models import MODEL_CONFIGS, MLP
from train import train_model, get_data_loaders, DEVICE, EPOCHS

os.makedirs("results", exist_ok=True)


def plot_curves(all_histories, configs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (name, history) in enumerate(all_histories.items()):
        ax = axes[idx]
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
        ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="tomato", linestyle="--")
        ax.set_title(configs[name]["label"], fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Loss Eğrileri — 4 Model Karşılaştırması", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/loss_curves.png", dpi=150)
    plt.close()
    print("  → results/loss_curves.png kaydedildi")


def plot_accuracy_curves(all_histories, configs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (name, history) in enumerate(all_histories.items()):
        ax = axes[idx]
        epochs = range(1, len(history["train_acc"]) + 1)
        ax.plot(epochs, [v * 100 for v in history["train_acc"]], label="Train Acc", color="steelblue")
        ax.plot(epochs, [v * 100 for v in history["val_acc"]],   label="Val Acc",   color="tomato", linestyle="--")
        ax.set_title(configs[name]["label"], fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(80, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Accuracy Eğrileri — 4 Model Karşılaştırması", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/accuracy_curves.png", dpi=150)
    plt.close()
    print("  → results/accuracy_curves.png kaydedildi")


def plot_confusion_matrix(model, test_loader, title):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            preds = model(images).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {title}", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.close()
    print("  → results/confusion_matrix.png kaydedildi")


def save_comparison_table(results):
    fieldnames = ["Model", "Mimari", "Optimizer", "LR", "Dropout",
                  "Son Train Acc (%)", "Son Val Acc (%)", "Test Acc (%)", "Eğitim Süresi (s)"]
    with open("results/comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print("  → results/comparison_table.csv kaydedildi")


def print_summary(results):
    print(f"\n{'='*80}")
    print(f"  SONUÇ ÖZETİ")
    print(f"{'='*80}")
    header = f"{'Model':<22} {'Test Acc':>9} {'Val Acc':>9} {'Train Acc':>10} {'Süre':>10}"
    print(header)
    print("-" * 65)
    for r in results:
        print(
            f"{r['Model']:<22} "
            f"{r['Test Acc (%)']:>8.2f}% "
            f"{r['Son Val Acc (%)']:>8.2f}% "
            f"{r['Son Train Acc (%)']:>9.2f}% "
            f"{r['Eğitim Süresi (s)']:>8.1f}s"
        )
    print("="*80)


def main():
    print("MNIST veri seti yükleniyor...")
    train_loader, val_loader, test_loader = get_data_loaders()

    all_histories = {}
    summary_rows = []
    best_name, best_acc, best_model = None, 0.0, None

    for name, config in MODEL_CONFIGS.items():
        history, test_acc, elapsed, model = train_model(
            name, config, train_loader, val_loader, test_loader, epochs=EPOCHS
        )
        all_histories[name] = history
        summary_rows.append({
            "Model": name,
            "Mimari": "→".join(["784"] + [str(h) for h in config["hidden_layers"]] + ["10"]),
            "Optimizer": config["optimizer"],
            "LR": config["lr"],
            "Dropout": "Evet" if config["use_dropout"] else "Hayır",
            "Son Train Acc (%)": round(history["train_acc"][-1] * 100, 2),
            "Son Val Acc (%)":   round(history["val_acc"][-1]   * 100, 2),
            "Test Acc (%)":      round(test_acc * 100, 2),
            "Eğitim Süresi (s)": round(elapsed, 1),
        })
        if test_acc > best_acc:
            best_acc, best_name, best_model = test_acc, name, model

    print("\nGrafikler oluşturuluyor...")
    plot_curves(all_histories, MODEL_CONFIGS)
    plot_accuracy_curves(all_histories, MODEL_CONFIGS)
    plot_confusion_matrix(best_model, test_loader, MODEL_CONFIGS[best_name]["label"])
    save_comparison_table(summary_rows)
    print_summary(summary_rows)
    print(f"\n  En iyi model: {best_name}  ({best_acc*100:.2f}% test accuracy)")
    print(f"  Tüm çıktılar: results/ klasörü\n")


if __name__ == "__main__":
    main()
