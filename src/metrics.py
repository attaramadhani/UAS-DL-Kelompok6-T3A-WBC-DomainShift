"""Metrik evaluasi per citra (Dice, IoU, mIoU, HD95) dan pengukuran latensi inferensi."""

import time
from typing import Dict, List
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _binary_dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> tuple[float, float]:
    """Hitung Dice dan IoU biner untuk satu kelas pada satu citra."""
    inter = float(np.logical_and(pred_mask, gt_mask).sum())
    pred_sum = float(pred_mask.sum())
    gt_sum = float(gt_mask.sum())
    union = pred_sum + gt_sum - inter
    if union == 0.0:
        return 1.0, 1.0
    dice = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)
    iou = (inter + eps) / (union + eps)
    return dice, iou


def _binary_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Hitung 95th percentile Hausdorff Distance (HD95) dalam piksel untuk satu citra."""
    pred_empty = not np.any(pred_mask)
    gt_empty = not np.any(gt_mask)
    if pred_empty and gt_empty:
        return 0.0
    if pred_empty or gt_empty:
        # Jarak penalti diagonal citra bila salah satu masker kosong total
        h, w = gt_mask.shape
        return float(np.hypot(h, w))

    struct = np.ones((3, 3), dtype=bool)
    pred_border = np.logical_xor(pred_mask, binary_erosion(pred_mask, structure=struct, border_value=0))
    gt_border = np.logical_xor(gt_mask, binary_erosion(gt_mask, structure=struct, border_value=0))

    # Gunakan seluruh piksel jika objek sangat kecil sehingga erosi menghapus tepi
    if not np.any(pred_border):
        pred_border = pred_mask
    if not np.any(gt_border):
        gt_border = gt_mask

    dt_gt = distance_transform_edt(~gt_border)
    dt_pred = distance_transform_edt(~pred_border)
    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]
    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def evaluate_single_image(pred: np.ndarray, gt: np.ndarray, compute_hd95: bool = True) -> Dict[str, float]:
    """Hitung seluruh metrik wajib untuk satu citra (0=bg, 1=sitoplasma, 2=nukleus)."""
    dice_cyto, iou_cyto = _binary_dice_iou(pred == 1, gt == 1)
    dice_nuc, iou_nuc = _binary_dice_iou(pred == 2, gt == 2)
    _, iou_bg = _binary_dice_iou(pred == 0, gt == 0)

    metrics = {
        "dice_cytoplasm": dice_cyto,
        "dice_nucleus": dice_nuc,
        "dice_fg_mean": 0.5 * (dice_cyto + dice_nuc),
        "iou_cytoplasm": iou_cyto,
        "iou_nucleus": iou_nuc,
        "miou_fg": 0.5 * (iou_cyto + iou_nuc),
        "miou_all": (iou_bg + iou_cyto + iou_nuc) / 3.0,
    }
    if compute_hd95:
        metrics["hd95_cytoplasm"] = _binary_hd95(pred == 1, gt == 1)
        metrics["hd95_nucleus"] = _binary_hd95(pred == 2, gt == 2)
    return metrics


def aggregate_per_image_metrics(per_image_rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Rangkum metrik per citra menjadi rerata dan simpangan baku (Kisi-Kisi Tabel 3.3)."""
    keys = [k for k in per_image_rows[0].keys() if isinstance(per_image_rows[0][k], (int, float))]
    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.array([r[k] for r in per_image_rows], dtype=np.float64)
        summary[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        }
    return summary


def benchmark_efficiency(model, input_size: int = 256, warmup: int = 10, n_iters: int = 50, device: str = "cuda"):
    """Ukur total parameter (juta) dan latensi batch=1 setelah warm-up (Kisi-Kisi K7)."""
    import torch

    model.eval().to(device)
    total_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    dummy = torch.randn(1, 3, input_size, input_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        latencies_ms = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "total_params_m": round(total_params_m, 3),
        "trainable_params_m": round(trainable_params_m, 3),
        "latency_mean_ms": round(float(arr.mean()), 3),
        "latency_std_ms": round(float(arr.std(ddof=1)), 3),
        "fps": round(1000.0 / float(arr.mean()), 2),
        "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "CPU",
    }
