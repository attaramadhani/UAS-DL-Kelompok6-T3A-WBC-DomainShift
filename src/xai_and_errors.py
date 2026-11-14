"""Analisis galeri kesalahan (min. 8 kasus) dan peta interpretabilitas Grad-CAM (Kisi-Kisi 3.4)."""

import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def summarize_error_categories(records: List[Dict]) -> Dict[str, int]:
    """Hitung frekuensi kategori kesalahan di seluruh set evaluasi yang diuji."""
    counts: Dict[str, int] = {}
    for r in records:
        cat = r["error_category"]
        counts[cat] = counts.get(cat, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def get_gradcam_target_layer(model):
    """Ambil layer konvolusi akhir encoder secara aman (ResNet, MobileNetV2, dll.)."""
    enc = getattr(model, "encoder", model)
    if hasattr(enc, "layer4"):
        return enc.layer4[-1]
    if hasattr(enc, "features"):
        return enc.features[-1]
    raise AttributeError(f"Tidak menemukan layer target Grad-CAM pada encoder {type(enc).__name__}")


def categorize_segmentation_error(pred: np.ndarray, gt: np.ndarray, img_rgb: np.ndarray, dice_fg: float) -> str:
    """Klasifikasikan penyebab utama galat segmentasi secara terukur."""
    gt_cyto = (gt == 1)
    pred_cyto = (pred == 1)
    gt_nuc = (gt == 2)
    pred_nuc = (pred == 2)

    nuc_union = np.logical_or(pred_nuc, gt_nuc).sum()
    nuc_iou = np.logical_and(pred_nuc, gt_nuc).sum() / max(nuc_union, 1)

    cyto_union = np.logical_or(pred_cyto, gt_cyto).sum()
    cyto_iou = np.logical_and(pred_cyto, gt_cyto).sum() / max(cyto_union, 1)

    fg_pred_area = (pred > 0).sum()
    fg_gt_area = (gt > 0).sum()

    if fg_gt_area == 0 or dice_fg < 0.35:
        return "Kegagalan deteksi / pergeseran domain berat"
    if fg_pred_area > 1.25 * fg_gt_area and cyto_iou < nuc_iou:
        return "Over-segmentasi sitoplasma (RBC/latar ikut tersegmentasi)"
    if fg_pred_area < 0.78 * fg_gt_area and cyto_iou < nuc_iou:
        return "Under-segmentasi sitoplasma (tepi pucat hilang)"
    if nuc_iou < cyto_iou:
        return "Kesalahan batas nukleus-sitoplasma"
    return "Ketidakakuratan kontur tepi ringan"


def save_error_gallery(records: List[Dict], out_path: str, top_k: int = 8) -> Dict[str, int]:
    """Simpan galeri minimal 8 contoh kesalahan terburuk beserta frekuensi kategorinya."""
    worst = sorted(records, key=lambda r: r["dice_fg_mean"])[:top_k]
    counts: Dict[str, int] = {}
    for r in worst:
        cat = r["error_category"]
        counts[cat] = counts.get(cat, 0) + 1

    cols = 3
    fig, axes = plt.subplots(len(worst), cols, figsize=(10, 3.0 * len(worst)))
    if len(worst) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, item in enumerate(worst):
        titles = [
            f"ID {item['id']} ({item['domain']})\nCitra Masukan",
            f"Ground Truth",
            f"Prediksi (Dice FG={item['dice_fg_mean']:.3f})\n{item['error_category']}",
        ]
        panels = [item["image_rgb"], item["gt_mask"], item["pred_mask"]]
        cmaps = [None, "viridis", "viridis"]
        for c in range(3):
            ax = axes[row_idx, c]
            if cmaps[c]:
                ax.imshow(panels[c], cmap=cmaps[c], vmin=0, vmax=2, interpolation="nearest")
            else:
                ax.imshow(panels[c])
            ax.set_title(titles[c], fontsize=8)
            ax.axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return counts


def generate_gradcam_grid(model, target_layer, samples: List[Dict], out_path: str, target_class: int = 2, device: str = "cuda"):
    """Buat visualisasi Grad-CAM untuk minimal 4 kasus benar dan 4 kasus salah (Kisi-Kisi 3.4)."""
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    class SemanticTarget:
        def __init__(self, category: int, mask_fn=None):
            self.category = category
            self.mask_fn = mask_fn

        def __call__(self, model_output):
            logits = model_output[0] if model_output.ndim == 4 else model_output
            if self.mask_fn is None:
                return logits[self.category, :, :].sum()
            return (logits[self.category, :, :] * self.mask_fn).sum()

    model.eval().to(device)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    cls_label = {0: "Background", 1: "Sitoplasma", 2: "Nukleus"}.get(target_class, f"Kelas {target_class}")

    fig, axes = plt.subplots(len(samples), 4, figsize=(12, 2.8 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for i, item in enumerate(samples):
            img_t = item["img_tensor"].unsqueeze(0).to(device)
            rgb_float = np.clip(item["img_tensor"].cpu().numpy().transpose(1, 2, 0) * std + mean, 0.0, 1.0).astype(np.float32)
            pred_cls = torch.from_numpy((item["pred_mask"] == target_class).astype(np.float32)).to(device)
            targets = [SemanticTarget(target_class, pred_cls if pred_cls.sum() > 0 else None)]
            grayscale_cam = cam(input_tensor=img_t, targets=targets)[0]
            cam_vis = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

            axes[i, 0].imshow(rgb_float)
            axes[i, 0].set_title(f"{item['tag']} | ID {item['id']}", fontsize=8)
            axes[i, 1].imshow(item["gt_mask"], cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
            axes[i, 1].set_title("Ground Truth", fontsize=8)
            axes[i, 2].imshow(item["pred_mask"], cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
            axes[i, 2].set_title(f"Prediksi (Dice={item['dice_fg_mean']:.3f})", fontsize=8)
            axes[i, 3].imshow(cam_vis)
            axes[i, 3].set_title(f"Grad-CAM ({cls_label})", fontsize=8)
            for c in range(4):
                axes[i, c].axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def records_to_dataframe(records: List[Dict], strategy: str = "", domain_label: str = ""):
    """Ekstrak metrik per citra dari records ke DataFrame bersih (tanpa tensor/array)."""
    import pandas as pd

    rows = []
    for r in records:
        rows.append({
            "id": r["id"],
            "domain": domain_label or r.get("domain", ""),
            "strategy": strategy,
            "gt_nuc_area_px": int(r.get("gt_nuc_area_px", 0)),
            "gt_cyto_area_px": int(r.get("gt_cyto_area_px", 0)),
            "gt_fg_area_px": int(r.get("gt_fg_area_px", 0)),
            "dice_nucleus": round(float(r.get("dice_nucleus", 0.0)), 4),
            "dice_cytoplasm": round(float(r.get("dice_cytoplasm", 0.0)), 4),
            "dice_fg_mean": round(float(r.get("dice_fg_mean", 0.0)), 4),
            "iou_nucleus": round(float(r.get("iou_nucleus", 0.0)), 4),
            "iou_cytoplasm": round(float(r.get("iou_cytoplasm", 0.0)), 4),
            "miou_fg": round(float(r.get("miou_fg", 0.0)), 4),
            "hd95_nucleus": round(float(r.get("hd95_nucleus", np.nan)), 2),
            "hd95_cytoplasm": round(float(r.get("hd95_cytoplasm", np.nan)), 2),
            "error_category": r.get("error_category", ""),
        })
    return pd.DataFrame(rows)


def summarize_by_object_size(records: List[Dict], strategy: str = "", domain_label: str = ""):
    """Kelompokkan performa per ukuran objek foreground (Kecil / Sedang / Besar, Kisi-Kisi K5)."""
    import pandas as pd

    df = records_to_dataframe(records, strategy=strategy, domain_label=domain_label)
    if df.empty or df["gt_fg_area_px"].nunique() < 3:
        df["size_bucket"] = "Semua"
    else:
        df["size_bucket"] = pd.qcut(df["gt_fg_area_px"], q=3, labels=["Kecil", "Sedang", "Besar"], duplicates="drop")

    rows = []
    for bucket, sub in df.groupby("size_bucket", observed=True):
        rows.append({
            "strategy": strategy,
            "domain": domain_label,
            "size_bucket": str(bucket),
            "n_images": len(sub),
            "fg_area_min_px": int(sub["gt_fg_area_px"].min()),
            "fg_area_max_px": int(sub["gt_fg_area_px"].max()),
            "dice_nucleus_mean": round(float(sub["dice_nucleus"].mean()), 4),
            "dice_cytoplasm_mean": round(float(sub["dice_cytoplasm"].mean()), 4),
            "miou_fg_mean": round(float(sub["miou_fg"].mean()), 4),
            "miou_fg_std": round(float(sub["miou_fg"].std(ddof=1)) if len(sub) > 1 else 0.0, 4),
            "hd95_nucleus_mean": round(float(sub["hd95_nucleus"].mean()), 2),
            "hd95_cytoplasm_mean": round(float(sub["hd95_cytoplasm"].mean()), 2),
        })
    return pd.DataFrame(rows)


def save_learning_curve_plot(curves: Dict[str, str], out_path: str, title: str = "Kurva Pelatihan & Validasi"):
    """Plot train_loss & val_dice_fg dari berkas CSV logs/histories/."""
    import pandas as pd

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for label, csv_path in curves.items():
        if not os.path.exists(csv_path):
            continue
        hdf = pd.read_csv(csv_path)
        axes[0].plot(hdf["epoch"], hdf["train_loss"], label=label, linewidth=2)
        axes[1].plot(hdf["epoch"], hdf["val_dice_fg"], label=label, linewidth=2)

    axes[0].set_title("Training Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Validation Dice Foreground per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Dice FG")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_baseline_preview_grid(
    in_records: List[Dict],
    cross_records: List[Dict],
    out_path: str,
    n_each: int = 2,
    first_label: str = "In-Domain (D1)",
    second_label: str = "Cross-Domain (D2)",
):
    """Tampilkan perbandingan visual prediksi dua kondisi evaluasi pada satu grid."""
    in_sorted = sorted(in_records, key=lambda r: r["dice_fg_mean"])
    cross_sorted = sorted(cross_records, key=lambda r: r["dice_fg_mean"])
    picks = []
    for tag, recs in [(first_label, in_sorted), (second_label, cross_sorted)]:
        if not recs:
            continue
        chosen = [recs[len(recs) - 1], recs[0]] if len(recs) > 1 else [recs[0]]
        for r in chosen[:n_each]:
            picks.append((tag, r))

    fig, axes = plt.subplots(len(picks), 3, figsize=(9.5, 2.8 * len(picks)))
    if len(picks) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (tag, item) in enumerate(picks):
        panels = [item["image_rgb"], item["gt_mask"], item["pred_mask"]]
        titles = [
            f"{tag}\nID {item['id']}",
            "Ground Truth",
            f"Prediksi (Dice={item['dice_fg_mean']:.3f} | mIoU={item.get('miou_fg', 0.0):.3f})",
        ]
        cmaps = [None, "viridis", "viridis"]
        for c in range(3):
            ax = axes[row_idx, c]
            if cmaps[c]:
                ax.imshow(panels[c], cmap=cmaps[c], vmin=0, vmax=2, interpolation="nearest")
            else:
                ax.imshow(panels[c])
            ax.set_title(titles[c], fontsize=8)
            ax.axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

