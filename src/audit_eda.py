"""Skrip Audit Data & EDA Resmi Milestone M2 pada dataset Zheng et al. (D1 & D2)."""

import json
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.dataset import (
    download_and_extract_zheng,
    encode_mask_to_classes,
    generate_fixed_splits,
    check_leakage_across_splits,
)
from src.stain_norm import MacenkoNormalizer, ReinhardNormalizer, select_representative_reference


def collect_domain_eda(domain_key: str, ddir: str) -> dict:
    """Audit ukuran citra, nilai piksel masker, distribusi kelas, dan statistik RGB."""
    bmps = sorted(glob.glob(os.path.join(ddir, "*.bmp")))
    sizes = {}
    stray_files = {}
    class_ratios = {0: [], 1: [], 2: []}
    rgb_means = []
    rgb_stds = []

    for bmp_p in bmps:
        stem = os.path.splitext(os.path.basename(bmp_p))[0]
        png_p = os.path.join(ddir, f"{stem}.png")
        img = np.asarray(Image.open(bmp_p).convert("RGB"))
        raw_mask = np.asarray(Image.open(png_p).convert("L"))

        sz = f"{img.shape[1]}x{img.shape[0]}"
        sizes[sz] = sizes.get(sz, 0) + 1

        extras = sorted(set(np.unique(raw_mask).tolist()) - {0, 128, 255})
        if extras:
            stray_files[os.path.basename(png_p)] = {int(v): int((raw_mask == v).sum()) for v in extras}

        cls_mask = encode_mask_to_classes(raw_mask)
        total_px = float(cls_mask.size)
        for c in (0, 1, 2):
            class_ratios[c].append(float((cls_mask == c).sum()) / total_px * 100.0)

        rgb_means.append(img.mean(axis=(0, 1)))
        rgb_stds.append(img.std(axis=(0, 1)))

    mean_arr = np.array(rgb_means)
    std_arr = np.array(rgb_stds)
    return {
        "domain": domain_key,
        "count": len(bmps),
        "unique_dimensions_count": len(sizes),
        "dimensions_summary": dict(sorted(sizes.items(), key=lambda x: -x[1])[:8]),
        "stray_mask_pixel_files": stray_files,
        "class_pixel_pct_mean_std": {
            "background": [round(float(np.mean(class_ratios[0])), 2), round(float(np.std(class_ratios[0])), 2)],
            "cytoplasm": [round(float(np.mean(class_ratios[1])), 2), round(float(np.std(class_ratios[1])), 2)],
            "nucleus": [round(float(np.mean(class_ratios[2])), 2), round(float(np.std(class_ratios[2])), 2)],
        },
        "rgb_channel_mean": [round(float(x), 2) for x in mean_arr.mean(axis=0)],
        "rgb_channel_std": [round(float(x), 2) for x in std_arr.mean(axis=0)],
        "_class_ratios_raw": class_ratios,
        "_rgb_means_raw": mean_arr,
    }


def make_overlay(img_rgb: np.ndarray, cls_mask: np.ndarray) -> np.ndarray:
    """Buat overlay transparan masker 3-kelas (sitoplasma=cyan, nukleus=merah/kuning)."""
    out = img_rgb.astype(np.float32).copy()
    cyto = cls_mask == 1
    nuc = cls_mask == 2
    out[cyto] = 0.55 * out[cyto] + 0.45 * np.array([0, 210, 255], dtype=np.float32)
    out[nuc] = 0.50 * out[nuc] + 0.50 * np.array([255, 60, 80], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_eda_figures(domain_dirs: dict, eda_d1: dict, eda_d2: dict, figs_dir: str) -> None:
    """Hasilkan 3 grafik audit data & normalisasi warna resolusi tinggi ke docs/figures/."""
    os.makedirs(figs_dir, exist_ok=True)

    # 1. Sampel citra + Ground Truth Overlay D1 vs D2
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for row_idx, (dkey, label) in enumerate([("D1", "Domain 1 (Mikroskop Biasa)"), ("D2", "Domain 2 (CellaVision)")]):
        bmps = sorted(glob.glob(os.path.join(domain_dirs[dkey], "*.bmp")))
        picks = [bmps[0], bmps[len(bmps) // 2]]
        for p_idx, bmp_p in enumerate(picks):
            stem = os.path.splitext(os.path.basename(bmp_p))[0]
            png_p = os.path.join(domain_dirs[dkey], f"{stem}.png")
            img = np.asarray(Image.open(bmp_p).convert("RGB"))
            cls_mask = encode_mask_to_classes(np.asarray(Image.open(png_p).convert("L")))
            ov = make_overlay(img, cls_mask)
            c0 = p_idx * 2
            axes[row_idx, c0].imshow(img)
            axes[row_idx, c0].set_title(f"{label}\nID {stem} ({img.shape[1]}x{img.shape[0]})", fontsize=8)
            axes[row_idx, c0].axis("off")
            axes[row_idx, c0 + 1].imshow(ov)
            axes[row_idx, c0 + 1].set_title(f"GT Mask Overlay\n(Merah=Nukleus, Biru=Sitoplasma)", fontsize=8)
            axes[row_idx, c0 + 1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fig1_sample_overlay_d1_d2.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Perbandingan distribusi luas piksel kelas & domain shift warna RGB
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    classes = ["Background", "Sitoplasma", "Nukleus"]
    d1_means = [np.mean(eda_d1["_class_ratios_raw"][c]) for c in (0, 1, 2)]
    d1_stds = [np.std(eda_d1["_class_ratios_raw"][c]) for c in (0, 1, 2)]
    d2_means = [np.mean(eda_d2["_class_ratios_raw"][c]) for c in (0, 1, 2)]
    d2_stds = [np.std(eda_d2["_class_ratios_raw"][c]) for c in (0, 1, 2)]
    x = np.arange(3)
    w = 0.35
    axes[0].bar(x - w / 2, d1_means, w, yerr=d1_stds, capsize=4, label="D1 (Mikroskop, n=300)", color="#E09F3E")
    axes[0].bar(x + w / 2, d2_means, w, yerr=d2_stds, capsize=4, label="D2 (CellaVision, n=100)", color="#6A4C93")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes)
    axes[0].set_ylabel("Proporsi Piksel per Citra (%)")
    axes[0].set_title("Distribusi Kelas Masker Ground Truth (D1 vs D2)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # RGB mean distributions
    channels = ["Red", "Green", "Blue"]
    d1_rgb = eda_d1["_rgb_means_raw"].mean(axis=0)
    d2_rgb = eda_d2["_rgb_means_raw"].mean(axis=0)
    axes[1].bar(x - w / 2, d1_rgb, w, label="D1 Mean RGB (Kekuningan)", color="#E09F3E")
    axes[1].bar(x + w / 2, d2_rgb, w, label="D2 Mean RGB (Keunguan)", color="#6A4C93")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channels)
    axes[1].set_ylabel("Intensitas Rerata Piksel (0-255)")
    axes[1].set_title("Bukti Kuantitatif Pergeseran Domain Warna (RGB Shift)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fig2_eda_distributions.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Preview Stain Normalization (D1 -> D2 train reference)
    import csv
    d1_imgs = sorted(glob.glob(os.path.join(domain_dirs["D1"], "*.bmp")))
    d2_split_csv = os.path.join(REPO_ROOT, "splits", "zheng_d2_split.csv")
    with open(d2_split_csv, encoding="utf-8") as f:
        d2_train_paths = [os.path.join(domain_dirs["D2"], r["image_file"]) for r in csv.DictReader(f) if r["split"] == "train"]
    ref_d2 = select_representative_reference(d2_train_paths)
    macenko = MacenkoNormalizer().fit(ref_d2)
    reinhard = ReinhardNormalizer().fit(ref_d2)

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    for r_idx, src_path in enumerate([d1_imgs[0], d1_imgs[5]]):
        src_rgb = np.asarray(Image.open(src_path).convert("RGB"))
        mac_rgb = macenko.transform(src_rgb)
        rei_rgb = reinhard.transform(src_rgb)
        panels = [
            (src_rgb, f"D1 Asli ({os.path.basename(src_path)})"),
            (ref_d2, "Referensi Target D2"),
            (mac_rgb, "Hasil Macenko Norm"),
            (rei_rgb, "Hasil Reinhard Norm"),
        ]
        for c_idx, (im, title) in enumerate(panels):
            axes[r_idx, c_idx].imshow(im)
            axes[r_idx, c_idx].set_title(title, fontsize=8)
            axes[r_idx, c_idx].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fig3_stain_norm_preview.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    data_root = os.path.join(REPO_ROOT, "data")
    splits_dir = os.path.join(REPO_ROOT, "splits")
    logs_dir = os.path.join(REPO_ROOT, "logs")
    figs_dir = os.path.join(REPO_ROOT, "docs", "figures")
    os.makedirs(logs_dir, exist_ok=True)

    domain_dirs = download_and_extract_zheng(data_root)
    split_csvs = generate_fixed_splits(domain_dirs, splits_dir, seed=42)
    leakage = check_leakage_across_splits(split_csvs, max_dhash_dist=4)

    eda_d1 = collect_domain_eda("D1", domain_dirs["D1"])
    eda_d2 = collect_domain_eda("D2", domain_dirs["D2"])
    generate_eda_figures(domain_dirs, eda_d1, eda_d2, figs_dir)

    report = {
        "dataset": "Zheng et al. (2018) WBC Segmentation (DOI: 10.17632/w7cvnmn4c5.1)",
        "split_seed": 42,
        "split_files": {k: os.path.relpath(v, REPO_ROOT).replace("\\", "/") for k, v in split_csvs.items()},
        "leakage_check": leakage,
        "domain_1_eda": {k: v for k, v in eda_d1.items() if not k.startswith("_")},
        "domain_2_eda": {k: v for k, v in eda_d2.items() if not k.startswith("_")},
    }
    report_path = os.path.join(logs_dir, "data_audit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Audit selesai! Laporan tersimpan di:", report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
