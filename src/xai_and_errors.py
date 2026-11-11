"""Analisis galeri kesalahan (min. 8 kasus) dan peta interpretabilitas Grad-CAM (Kisi-Kisi 3.4)."""

from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


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

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
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

    fig, axes = plt.subplots(len(samples), 4, figsize=(12, 2.8 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for i, item in enumerate(samples):
            img_t = item["img_tensor"].unsqueeze(0).to(device)
            rgb_float = np.clip(item["img_tensor"].cpu().numpy().transpose(1, 2, 0) * std + mean, 0.0, 1.0)
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
            axes[i, 3].set_title(f"Grad-CAM (Kelas {target_class})", fontsize=8)
            for c in range(4):
                axes[i, c].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
