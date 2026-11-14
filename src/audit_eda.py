"""Skrip Audit Data & EDA Resmi Milestone M2 pada dataset Zheng et al. (D1 & D2)."""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label as cc_label

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset import (
    check_leakage_across_splits,
    download_and_extract_zheng,
    encode_mask_to_classes,
    generate_fixed_splits,
)
from src.stain_norm import MacenkoNormalizer, ReinhardNormalizer, select_representative_reference


def _quantile_stats(arr: np.ndarray, digits: int = 2) -> Dict[str, float]:
    """Hitung statistik sebaran lengkap (min, Q1, median, Q3, max, mean, std)."""
    return {
        "min": round(float(np.min(arr)), digits),
        "p25": round(float(np.percentile(arr, 25)), digits),
        "median": round(float(np.median(arr)), digits),
        "p75": round(float(np.percentile(arr, 75)), digits),
        "max": round(float(np.max(arr)), digits),
        "mean": round(float(np.mean(arr)), digits),
        "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, digits),
    }


def collect_domain_eda(
    domain_key: str,
    ddir: str,
    split_csv_path: str | None = None,
) -> dict:
    """Audit lengkap ukuran citra, masker mentah/terkonversi, jumlah & luas objek, serta RGB."""
    split_lookup: Dict[str, Dict[str, str]] = {}
    if split_csv_path and os.path.exists(split_csv_path):
        with open(split_csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                split_lookup[row["id"]] = row

    bmps = sorted(glob.glob(os.path.join(ddir, "*.bmp")))
    sizes: Dict[str, int] = {}
    modes: Dict[str, int] = {}
    stray_files: Dict[str, Dict[int, int]] = {}
    global_raw_vals = set()
    missing_masks: List[str] = []
    corrupt_images: List[str] = []

    class_ratios = {0: [], 1: [], 2: []}
    area_lists = {"cyto_px": [], "nuc_px": [], "fg_px": [], "nc_ratio": []}
    comp_lists = {"nuc_comp": [], "cyto_comp": [], "fg_comp": []}
    file_kbs: List[float] = []
    rgb_means: List[np.ndarray] = []
    rgb_stds: List[np.ndarray] = []
    per_image_rows: List[Dict[str, object]] = []

    conn8 = np.ones((3, 3), dtype=np.int32)

    for bmp_p in bmps:
        stem = os.path.splitext(os.path.basename(bmp_p))[0]
        png_p = os.path.join(ddir, f"{stem}.png")
        if not os.path.exists(png_p):
            missing_masks.append(os.path.basename(bmp_p))
            continue

        try:
            with Image.open(bmp_p) as im_raw:
                modes[im_raw.mode] = modes.get(im_raw.mode, 0) + 1
                img = np.asarray(im_raw.convert("RGB"))
            with Image.open(png_p) as mk_raw:
                raw_mask = np.asarray(mk_raw.convert("L"))
        except Exception as exc:
            corrupt_images.append(f"{os.path.basename(bmp_p)}: {exc}")
            continue

        h, w = img.shape[:2]
        sz = f"{w}x{h}"
        sizes[sz] = sizes.get(sz, 0) + 1
        kb = os.path.getsize(bmp_p) / 1024.0
        file_kbs.append(kb)

        raw_unique = sorted(int(v) for v in np.unique(raw_mask).tolist())
        global_raw_vals.update(raw_unique)
        extras = sorted(set(raw_unique) - {0, 128, 255})
        stray_counts = {int(v): int((raw_mask == v).sum()) for v in extras}
        if extras:
            stray_files[os.path.basename(png_p)] = stray_counts

        cls_mask = encode_mask_to_classes(raw_mask)
        total_px = float(cls_mask.size)
        bg_px = int((cls_mask == 0).sum())
        cyto_px = int((cls_mask == 1).sum())
        nuc_px = int((cls_mask == 2).sum())
        fg_px = cyto_px + nuc_px

        bg_pct = bg_px / total_px * 100.0
        cyto_pct = cyto_px / total_px * 100.0
        nuc_pct = nuc_px / total_px * 100.0
        fg_pct = fg_px / total_px * 100.0
        nc_ratio = nuc_px / max(cyto_px, 1)

        _, n_nuc = cc_label(cls_mask == 2, structure=conn8)
        _, n_cyto = cc_label(cls_mask == 1, structure=conn8)
        _, n_fg = cc_label(cls_mask > 0, structure=conn8)

        class_ratios[0].append(bg_pct)
        class_ratios[1].append(cyto_pct)
        class_ratios[2].append(nuc_pct)
        area_lists["cyto_px"].append(cyto_px)
        area_lists["nuc_px"].append(nuc_px)
        area_lists["fg_px"].append(fg_px)
        area_lists["nc_ratio"].append(nc_ratio)
        comp_lists["nuc_comp"].append(int(n_nuc))
        comp_lists["cyto_comp"].append(int(n_cyto))
        comp_lists["fg_comp"].append(int(n_fg))

        im_mean = img.mean(axis=(0, 1))
        im_std = img.std(axis=(0, 1))
        rgb_means.append(im_mean)
        rgb_stds.append(im_std)

        sp_meta = split_lookup.get(stem, {})
        per_image_rows.append({
            "domain": domain_key,
            "id": stem,
            "split": sp_meta.get("split", "unknown"),
            "image_file": os.path.basename(bmp_p),
            "mask_file": os.path.basename(png_p),
            "width": w,
            "height": h,
            "aspect_ratio": round(w / max(h, 1), 3),
            "file_size_kb": round(kb, 2),
            "raw_mask_values": ",".join(str(v) for v in raw_unique),
            "has_stray_pixels": bool(extras),
            "stray_pixel_total": sum(stray_counts.values()),
            "bg_pct": round(bg_pct, 2),
            "cyto_pct": round(cyto_pct, 2),
            "nuc_pct": round(nuc_pct, 2),
            "fg_pct": round(fg_pct, 2),
            "cyto_area_px": cyto_px,
            "nuc_area_px": nuc_px,
            "fg_area_px": fg_px,
            "nc_ratio": round(nc_ratio, 4),
            "nuc_components": int(n_nuc),
            "cyto_components": int(n_cyto),
            "fg_components": int(n_fg),
            "rgb_mean_r": round(float(im_mean[0]), 2),
            "rgb_mean_g": round(float(im_mean[1]), 2),
            "rgb_mean_b": round(float(im_mean[2]), 2),
            "rgb_std_r": round(float(im_std[0]), 2),
            "rgb_std_g": round(float(im_std[1]), 2),
            "rgb_std_b": round(float(im_std[2]), 2),
            "dhash": sp_meta.get("dhash", ""),
            "sha256": sp_meta.get("sha256", ""),
        })

    mean_arr = np.array(rgb_means)
    std_arr = np.array(rgb_stds)
    sorted_dims = dict(sorted(sizes.items(), key=lambda x: -x[1]))

    return {
        "domain": domain_key,
        "count": len(per_image_rows),
        "missing_masks_count": len(missing_masks),
        "corrupt_images_count": len(corrupt_images),
        "image_modes": modes,
        "file_size_kb_stats": {**_quantile_stats(np.array(file_kbs), 2), "total_mb": round(sum(file_kbs) / 1024.0, 2)},
        "unique_dimensions_count": len(sorted_dims),
        "dimensions_all": sorted_dims,
        "dimensions_summary": sorted_dims,
        "raw_mask_unique_values_global": sorted(global_raw_vals),
        "stray_files_count": len(stray_files),
        "stray_mask_pixel_files": stray_files,
        "class_pixel_pct_mean_std": {
            "background": [round(float(np.mean(class_ratios[0])), 2), round(float(np.std(class_ratios[0])), 2)],
            "cytoplasm": [round(float(np.mean(class_ratios[1])), 2), round(float(np.std(class_ratios[1])), 2)],
            "nucleus": [round(float(np.mean(class_ratios[2])), 2), round(float(np.std(class_ratios[2])), 2)],
        },
        "class_pixel_pct_stats": {
            "background": _quantile_stats(np.array(class_ratios[0]), 2),
            "cytoplasm": _quantile_stats(np.array(class_ratios[1]), 2),
            "nucleus": _quantile_stats(np.array(class_ratios[2]), 2),
        },
        "object_area_px_stats": {
            "cytoplasm": _quantile_stats(np.array(area_lists["cyto_px"]), 1),
            "nucleus": _quantile_stats(np.array(area_lists["nuc_px"]), 1),
            "foreground_total": _quantile_stats(np.array(area_lists["fg_px"]), 1),
            "nc_ratio": _quantile_stats(np.array(area_lists["nc_ratio"]), 3),
        },
        "object_count_per_image": {
            "cell_foreground_components": _quantile_stats(np.array(comp_lists["fg_comp"]), 2),
            "nucleus_components": _quantile_stats(np.array(comp_lists["nuc_comp"]), 2),
            "cytoplasm_components": _quantile_stats(np.array(comp_lists["cyto_comp"]), 2),
        },
        "rgb_channel_mean": [round(float(x), 2) for x in mean_arr.mean(axis=0)],
        "rgb_channel_std": [round(float(x), 2) for x in std_arr.mean(axis=0)],
        "rgb_channel_median": [round(float(x), 2) for x in np.median(mean_arr, axis=0)],
        "rgb_channel_min": [round(float(x), 2) for x in mean_arr.min(axis=0)],
        "rgb_channel_max": [round(float(x), 2) for x in mean_arr.max(axis=0)],
        "_class_ratios_raw": class_ratios,
        "_rgb_means_raw": mean_arr,
        "_area_lists_raw": area_lists,
        "_dims_raw": sorted_dims,
        "_per_image_rows": per_image_rows,
    }


def make_overlay(img_rgb: np.ndarray, cls_mask: np.ndarray) -> np.ndarray:
    """Buat overlay transparan masker 3-kelas (sitoplasma=cyan, nukleus=merah)."""
    out = img_rgb.astype(np.float32).copy()
    cyto = cls_mask == 1
    nuc = cls_mask == 2
    out[cyto] = 0.55 * out[cyto] + 0.45 * np.array([0, 210, 255], dtype=np.float32)
    out[nuc] = 0.50 * out[nuc] + 0.50 * np.array([255, 60, 80], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_mask_conversion_audit_24(domain_dirs: dict, df_audit: pd.DataFrame, figs_dir: str) -> None:
    """Buat grid verifikasi visual 24 citra (12 D1 + 12 D2 lintas train/val/test, min. 20 sesuai Kisi-Kisi 3.1)."""
    picked_rows = []
    for dkey in ("D1", "D2"):
        sub = df_audit[df_audit["domain"] == dkey]
        for sp_name, n_pick in [("train", 4), ("val", 4), ("test", 4)]:
            sp_sub = sub[sub["split"] == sp_name].reset_index(drop=True)
            if len(sp_sub) == 0:
                continue
            idxs = np.linspace(0, len(sp_sub) - 1, min(n_pick, len(sp_sub)), dtype=int)
            for idx in idxs:
                picked_rows.append(sp_sub.iloc[int(idx)])

    n_items = len(picked_rows)
    ncols = 6
    nrows = int(np.ceil(n_items / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.5, 2.9 * nrows))
    axes = np.atleast_2d(axes)

    for idx, r in enumerate(picked_rows):
        ax = axes[idx // ncols, idx % ncols]
        ddir = domain_dirs[r["domain"]]
        img = np.asarray(Image.open(os.path.join(ddir, r["image_file"])).convert("RGB"))
        raw_mask = np.asarray(Image.open(os.path.join(ddir, r["mask_file"])).convert("L"))
        cls_mask = encode_mask_to_classes(raw_mask)
        ov = make_overlay(img, cls_mask)
        ax.imshow(ov)
        stray_tag = " | stray" if r["has_stray_pixels"] else ""
        ax.set_title(
            f"{r['domain']} [{r['split']}] ID {r['id']}\n"
            f"{r['width']}x{r['height']} | N:{r['nuc_pct']:.1f}% C:{r['cyto_pct']:.1f}%{stray_tag}",
            fontsize=7.5,
        )
        ax.axis("off")

    for idx in range(n_items, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle(
        "Verifikasi Visual Konversi Masker 3-Kelas pada 24 Citra (12 D1 + 12 D2, Merah=Nukleus, Cyan=Sitoplasma)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fig1b_mask_conversion_audit_24.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_eda_figures(domain_dirs: dict, eda_d1: dict, eda_d2: dict, df_audit: pd.DataFrame, figs_dir: str) -> None:
    """Hasilkan seluruh grafik audit data & normalisasi warna resolusi tinggi ke docs/figures/."""
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
            axes[row_idx, c0 + 1].set_title("GT Mask Overlay\n(Merah=Nukleus, Cyan=Sitoplasma)", fontsize=8)
            axes[row_idx, c0 + 1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fig1_sample_overlay_d1_d2.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 1b. Pemeriksaan visual konversi anotasi pada 24 citra (Kisi-Kisi 3.1 min. 20 citra)
    _save_mask_conversion_audit_24(domain_dirs, df_audit, figs_dir)

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

    # 2b. Sebaran ukuran objek (luas piksel), rasio N:C, & dimensi citra D1 vs D2
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    d1_sub = df_audit[df_audit["domain"] == "D1"]
    d2_sub = df_audit[df_audit["domain"] == "D2"]

    bp1 = axes[0].boxplot(
        [d1_sub["nuc_pct"], d2_sub["nuc_pct"], d1_sub["cyto_pct"], d2_sub["cyto_pct"]],
        patch_artist=True,
    )
    axes[0].set_xticklabels(["D1 Nukleus%", "D2 Nukleus%", "D1 Sito%", "D2 Sito%"])
    for patch, clr in zip(bp1["boxes"], ["#E09F3E", "#6A4C93", "#F4C95D", "#9B7EDE"]):
        patch.set_facecolor(clr)
        patch.set_alpha(0.75)
    axes[0].set_ylabel("Luas Relatif terhadap Citra (%)")
    axes[0].set_title("Sebaran Luas Objek Nukleus & Sitoplasma")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(d1_sub["nc_ratio"], bins=18, alpha=0.65, color="#E09F3E", label="D1 (Mikroskop)")
    axes[1].hist(d2_sub["nc_ratio"], bins=18, alpha=0.65, color="#6A4C93", label="D2 (CellaVision)")
    axes[1].set_xlabel("Rasio Luas Nukleus / Sitoplasma")
    axes[1].set_ylabel("Jumlah Citra")
    axes[1].set_title("Sebaran Rasio N:C per Citra")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    dim_rows = []
    for dkey, color, take in [("D1", "#E09F3E", 8), ("D2", "#6A4C93", 4)]:
        dims = eda_d1["_dims_raw"] if dkey == "D1" else eda_d2["_dims_raw"]
        for k, v in list(dims.items())[:take]:
            dim_rows.append((f"{dkey}: {k}", v, color))
    dim_rows.sort(key=lambda r: -r[1])
    y_pos = np.arange(len(dim_rows))
    axes[2].barh(y_pos, [r[1] for r in dim_rows], color=[r[2] for r in dim_rows])
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels([r[0] for r in dim_rows], fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Jumlah Citra")
    axes[2].set_title("Resolusi Asli Terbanyak (D1 vs D2)")
    axes[2].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fig2b_object_size_and_nc_distributions.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Preview Stain Normalization (D1 -> D2 train reference)
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


def build_audit_summary_tables(report: dict, df_audit: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Susun seluruh tabel audit dataset siap tampil (domain, split, kebocoran, dimensi, sebaran objek)."""
    d1, d2 = report["domain_1_eda"], report["domain_2_eda"]
    leak = report["leakage_check"]

    domain_rows = []
    for d in (d1, d2):
        domain_rows.append({
            "Domain": d["domain"],
            "Jumlah_Citra": d["count"],
            "Masker_Hilang": d["missing_masks_count"],
            "Citra_Rusak": d["corrupt_images_count"],
            "Dimensi_Unik": d["unique_dimensions_count"],
            "Total_Size_MB": d["file_size_kb_stats"]["total_mb"],
            "Nilai_Masker_Mentah": str(d["raw_mask_unique_values_global"]),
            "File_Stray_Pixel": d["stray_files_count"],
            "RGB_Mean_(R,G,B)": str(d["rgb_channel_mean"]),
            "RGB_Std_(R,G,B)": str(d["rgb_channel_std"]),
        })

    split_rows = []
    for dkey in ("D1", "D2"):
        counts = leak["split_counts"].get(dkey, {})
        tot = sum(counts.values()) or 1
        split_rows.append({
            "Domain": dkey,
            "Train": counts.get("train", 0),
            "Val": counts.get("val", 0),
            "Test": counts.get("test", 0),
            "Total": sum(counts.values()),
            "Proporsi_(Train/Val/Test_%)" : (
                f"{counts.get('train', 0)/tot*100:.1f}% / "
                f"{counts.get('val', 0)/tot*100:.1f}% / "
                f"{counts.get('test', 0)/tot*100:.1f}%"
            ),
        })

    leak_rows = [
        {"Pemeriksaan": "Total Citra D1 + D2", "Hasil": leak["total_images"], "Keterangan": "Gabungan Dataset 1 + Dataset 2"},
        {"Pemeriksaan": "Kebocoran Exact (SHA-256) Lintas Split", "Hasil": leak["cross_split_leakage_exact"], "Keterangan": str(leak.get("cross_split_leakage_exact_pairs", []))},
        {"Pemeriksaan": "Kebocoran Near-Duplicate (dHash <= 4 bit) Lintas Split", "Hasil": leak["cross_split_leakage_dhash_le_4bits"], "Keterangan": str(leak.get("cross_split_leakage_dhash_pairs", []))},
        {"Pemeriksaan": "Near-Duplicate Lintas Domain (dHash <= 4 bit)", "Hasil": leak["cross_domain_near_duplicates_le_4bits"], "Keterangan": str(leak.get("cross_domain_near_duplicate_pairs", []))},
        {"Pemeriksaan": "Grup Duplikat Identik Dikunci di Set Train", "Hasil": len(leak["within_train_exact_duplicates_grouped"]), "Keterangan": str(leak["within_train_exact_duplicates_grouped"])},
    ]

    dim_rows = []
    for d in (d1, d2):
        for dim_str, cnt in d["dimensions_all"].items():
            dim_rows.append({
                "Domain": d["domain"],
                "Dimensi_(WxH)": dim_str,
                "Jumlah_Citra": cnt,
                "Persentase_(%)": round(cnt / max(d["count"], 1) * 100.0, 2),
            })

    obj_rows = []
    for dkey in ("D1", "D2"):
        sub = df_audit[df_audit["domain"] == dkey]
        for label, col in [
            ("Proporsi Background (%)", "bg_pct"),
            ("Proporsi Sitoplasma (%)", "cyto_pct"),
            ("Proporsi Nukleus (%)", "nuc_pct"),
            ("Luas Nukleus (piksel)", "nuc_area_px"),
            ("Luas Sitoplasma (piksel)", "cyto_area_px"),
            ("Luas Foreground Sel (piksel)", "fg_area_px"),
            ("Rasio N:C (Nukleus/Sitoplasma)", "nc_ratio"),
            ("Jumlah Komponen Nukleus/Citra", "nuc_components"),
            ("Jumlah Komponen Sel Foreground/Citra", "fg_components"),
        ]:
            s = _quantile_stats(sub[col].to_numpy(dtype=np.float64), 2)
            obj_rows.append({
                "Domain": dkey,
                "Metrik_Objek": label,
                "Mean": s["mean"],
                "Std": s["std"],
                "Min": s["min"],
                "Q1_(P25)": s["p25"],
                "Median": s["median"],
                "Q3_(P75)": s["p75"],
                "Max": s["max"],
            })

    stray_df = df_audit[df_audit["has_stray_pixels"]][
        ["domain", "id", "split", "image_file", "mask_file", "raw_mask_values", "stray_pixel_total"]
    ].reset_index(drop=True)

    return {
        "domain_overview": pd.DataFrame(domain_rows),
        "split_composition": pd.DataFrame(split_rows),
        "leakage_report": pd.DataFrame(leak_rows),
        "dimensions_complete": pd.DataFrame(dim_rows),
        "object_statistics": pd.DataFrame(obj_rows),
        "stray_mask_files": stray_df,
    }


def main():
    data_root = os.path.join(REPO_ROOT, "data")
    splits_dir = os.path.join(REPO_ROOT, "splits")
    logs_dir = os.path.join(REPO_ROOT, "logs")
    figs_dir = os.path.join(REPO_ROOT, "docs", "figures")
    os.makedirs(logs_dir, exist_ok=True)

    domain_dirs = download_and_extract_zheng(data_root)
    split_csvs = generate_fixed_splits(domain_dirs, splits_dir, seed=42)
    leakage = check_leakage_across_splits(split_csvs, max_dhash_dist=4)

    eda_d1 = collect_domain_eda("D1", domain_dirs["D1"], split_csvs["D1"])
    eda_d2 = collect_domain_eda("D2", domain_dirs["D2"], split_csvs["D2"])

    all_per_image = eda_d1["_per_image_rows"] + eda_d2["_per_image_rows"]
    df_audit = pd.DataFrame(all_per_image)
    per_image_csv_path = os.path.join(logs_dir, "data_audit_per_image.csv")
    df_audit.to_csv(per_image_csv_path, index=False)

    generate_eda_figures(domain_dirs, eda_d1, eda_d2, df_audit, figs_dir)

    report = {
        "dataset": "Zheng et al. (2018) WBC Segmentation (DOI: 10.17632/w7cvnmn4c5.1)",
        "split_seed": 42,
        "split_files": {k: os.path.relpath(v, REPO_ROOT).replace("\\", "/") for k, v in split_csvs.items()},
        "per_image_audit_csv": os.path.relpath(per_image_csv_path, REPO_ROOT).replace("\\", "/"),
        "leakage_check": leakage,
        "domain_1_eda": {k: v for k, v in eda_d1.items() if not k.startswith("_")},
        "domain_2_eda": {k: v for k, v in eda_d2.items() if not k.startswith("_")},
    }
    report_path = os.path.join(logs_dir, "data_audit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    tables = build_audit_summary_tables(report, df_audit)
    print(f"Audit lengkap selesai! Tersimpan di:\n- {report_path}\n- {per_image_csv_path}")
    return {
        "report": report,
        "report_path": report_path,
        "per_image_csv_path": per_image_csv_path,
        "per_image_df": df_audit,
        "tables": tables,
    }


if __name__ == "__main__":
    out = main()
    for name, tbl in out["tables"].items():
        print(f"\n=== {name.upper()} ===")
        print(tbl.to_string(index=False))
