"""Fungsi agregasi tabel, uji statistik, dan grafik laporan untuk Milestone M3."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


CORE_METRICS = [
    "in_dice_nuc", "in_dice_cyto", "in_iou_nuc", "in_iou_cyto", "in_miou_fg", "in_hd95_nuc", "in_hd95_cyto",
    "cross_dice_nuc", "cross_dice_cyto", "cross_iou_nuc", "cross_iou_cyto", "cross_miou_fg", "cross_hd95_nuc", "cross_hd95_cyto",
]


def build_main_summary(df_runs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rangkum eksperimen utama 3-seed menjadi CSV datar (flat) dan tabel format mean+-std."""
    df_main = df_runs[df_runs["run_name"].str.startswith("Main_")].copy()
    if "status" in df_main.columns:
        df_main = df_main[df_main["status"] == "success"]
    df_main["delta_miou_drop"] = df_main["in_miou_fg"] - df_main["cross_miou_fg"]

    cols = [c for c in CORE_METRICS if c in df_main.columns] + ["delta_miou_drop"]
    group_cols = ["train_domain", "target_domain", "arch", "strategy"]
    grouped = df_main.groupby(group_cols, as_index=False)

    flat_rows: List[Dict[str, object]] = []
    fmt_rows: List[Dict[str, object]] = []
    for key, sub in grouped:
        tdom, xdom, arch, strat = key
        flat_r: Dict[str, object] = {
            "train_domain": tdom,
            "target_domain": xdom,
            "direction": f"{tdom} -> {xdom}",
            "arch": arch,
            "strategy": strat,
            "n_seeds": len(sub),
        }
        fmt_r: Dict[str, object] = {
            "Arah": f"{tdom} -> {xdom}",
            "Arsitektur": arch,
            "Strategi": strat,
            "N": len(sub),
        }
        for m in cols:
            vals = sub[m].astype(float)
            mu = float(vals.mean())
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            digits = 2 if "hd95" in m else 4
            flat_r[f"{m}_mean"] = round(mu, digits)
            flat_r[f"{m}_std"] = round(sd, digits)
            fmt_r[m] = f"{mu:.{digits}f} ± {sd:.{digits}f}"
        flat_rows.append(flat_r)
        fmt_rows.append(fmt_r)

    return pd.DataFrame(flat_rows), pd.DataFrame(fmt_rows)


def build_statistical_tests(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Uji perbandingan berpasangan antarstrategi (vs NoAdapt, n=3 seed) & antararsitektur (n=12)."""
    df_main = df_runs[df_runs["run_name"].str.startswith("Main_")].copy()
    if "status" in df_main.columns:
        df_main = df_main[df_main["status"] == "success"]

    rows: List[Dict[str, object]] = []
    for (tdom, xdom, arch), sub in df_main.groupby(["train_domain", "target_domain", "arch"]):
        base = sub[sub["strategy"] == "NoAdapt"].set_index("seed")["cross_miou_fg"]
        for strat in ("ColorJitter", "Macenko", "Reinhard"):
            cand = sub[sub["strategy"] == strat].set_index("seed")["cross_miou_fg"]
            common = base.index.intersection(cand.index)
            if len(common) < 2:
                continue
            diffs = cand.loc[common].to_numpy(dtype=float) - base.loc[common].to_numpy(dtype=float)
            t_stat, p_val = stats.ttest_rel(cand.loc[common], base.loc[common])
            rows.append({
                "comparison": f"{tdom}->{xdom} | {arch}: {strat} vs NoAdapt",
                "n_pairs": len(common),
                "mean_baseline": round(float(base.loc[common].mean()), 4),
                "mean_candidate": round(float(cand.loc[common].mean()), 4),
                "mean_diff": round(float(diffs.mean()), 4),
                "std_diff": round(float(diffs.std(ddof=1)), 4),
                "t_stat": round(float(t_stat), 4),
                "p_ttest_paired": round(float(p_val), 5),
                "p_wilcoxon": np.nan,
            })

    # Perbandingan paradigma: DeepLabV3+ vs U-Net berpasangan pada (arah, strategi, seed)
    piv = df_main.pivot_table(
        index=["train_domain", "target_domain", "strategy", "seed"],
        columns="arch",
        values="cross_miou_fg",
    ).dropna()
    if {"deeplabv3plus", "unet"}.issubset(piv.columns) and len(piv) >= 4:
        d_vals = piv["deeplabv3plus"].to_numpy(dtype=float)
        u_vals = piv["unet"].to_numpy(dtype=float)
        t_stat, p_t = stats.ttest_rel(d_vals, u_vals)
        try:
            _, p_w = stats.wilcoxon(d_vals, u_vals)
        except Exception:
            p_w = np.nan
        rows.append({
            "comparison": "Semua Kondisi: DeepLabV3+ vs U-Net",
            "n_pairs": len(piv),
            "mean_baseline": round(float(u_vals.mean()), 4),
            "mean_candidate": round(float(d_vals.mean()), 4),
            "mean_diff": round(float((d_vals - u_vals).mean()), 4),
            "std_diff": round(float((d_vals - u_vals).std(ddof=1)), 4),
            "t_stat": round(float(t_stat), 4),
            "p_ttest_paired": round(float(p_t), 5),
            "p_wilcoxon": round(float(p_w), 5) if not np.isnan(p_w) else np.nan,
        })

    return pd.DataFrame(rows)


def build_ablation_tables(df_runs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Susun tabel ablasi satu-faktor rapi untuk seluruh 4 anggota tim + referensi kanonik."""
    def _pick(run_names: List[str], member: str, topic: str, ref_name: str) -> pd.DataFrame:
        sub = df_runs[df_runs["run_name"].isin(run_names)].copy()
        if sub.empty:
            return sub
        order_map = {n: i for i, n in enumerate(run_names)}
        sub["_ord"] = sub["run_name"].map(order_map)
        sub = sub.sort_values("_ord").drop(columns=["_ord"])
        ref_rows = sub[sub["run_name"] == ref_name]
        ref_miou = float(ref_rows["cross_miou_fg"].iloc[0]) if not ref_rows.empty else float(sub["cross_miou_fg"].iloc[0])
        sub.insert(0, "member", member)
        sub.insert(1, "ablation_topic", topic)
        sub["delta_cross_miou_vs_ref"] = (sub["cross_miou_fg"] - ref_miou).round(4)
        return sub

    attala = _pick(
        [
            "Main_unet_NoAdapt_D1_to_D2_s42",
            "Main_unet_ColorJitter_D1_to_D2_s42",
            "Main_unet_Macenko_D1_to_D2_s42",
            "Abl_Attala_Macenko_EvalOnly",
            "Main_unet_Reinhard_D1_to_D2_s42",
            "Abl_Attala_Reinhard_EvalOnly",
        ],
        member="Attala (230441100144)",
        topic="Strategi Warna & Train+Eval vs Eval-Only (U-Net D1->D2, s42)",
        ref_name="Main_unet_NoAdapt_D1_to_D2_s42",
    )
    nafaul = _pick(
        ["Abl_Nafaul_Loss_ce", "Abl_Nafaul_Loss_dice", "Abl_Nafaul_Loss_ce_dice"],
        member="Nafaul (240441100125)",
        topic="Fungsi Loss: CE vs Dice vs CE+Dice (U-Net D1->D2, s42)",
        ref_name="Abl_Nafaul_Loss_ce_dice",
    )
    rafly = _pick(
        ["Abl_Rafly_Sz128_geom", "Abl_Rafly_Sz256_none", "Abl_Rafly_Sz256_geom"],
        member="M. Rafly (240441100086)",
        topic="Resolusi & Augmentasi Geometrik (DeepLabV3+ D1->D2, s42)",
        ref_name="Abl_Rafly_Sz256_geom",
    )
    naufal = _pick(
        ["Abl_Naufal_resnet34_Full", "Abl_Naufal_mobilenet_v2_Full", "Abl_Naufal_resnet34_Frozen"],
        member="Naufal (240441100038)",
        topic="Encoder & Strategi Fine-Tuning (DeepLabV3+ D1->D2, s42)",
        ref_name="Abl_Naufal_resnet34_Full",
    )

    per_member = {"Attala": attala, "Nafaul": nafaul, "M_Rafly": rafly, "Naufal": naufal}
    non_empty = [t for t in per_member.values() if not t.empty]
    combined = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    return combined, per_member


def build_efficiency_table(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Bangun tabel efisiensi model K7 (parameter total/trainable, latensi mean+-std, FPS, GPU)."""
    cols = [
        "run_name", "arch", "encoder", "freeze_encoder", "input_size",
        "params_m", "trainable_params_m", "latency_ms", "latency_std_ms", "fps", "gpu", "cross_miou_fg",
    ]
    sub = df_runs[df_runs["seed"] == 42].copy()
    sub = sub.drop_duplicates(subset=["arch", "encoder", "freeze_encoder", "input_size"])
    avail = [c for c in cols if c in sub.columns]
    return sub[avail].reset_index(drop=True)


def save_domain_shift_bar_figure(df_runs: pd.DataFrame, out_path: str) -> None:
    """Plot perbandingan Cross-Domain mIoU 3-seed mean+-std untuk D1->D2 dan D2->D1."""
    df_main = df_runs[df_runs["run_name"].str.startswith("Main_")].copy()
    df_main["direction"] = df_main["train_domain"] + " -> " + df_main["target_domain"]

    strategies_order = ["NoAdapt", "ColorJitter", "Macenko", "Reinhard"]
    colors = {"NoAdapt": "#94A3B8", "ColorJitter": "#F59E0B", "Macenko": "#2563EB", "Reinhard": "#10B981"}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    for ax, direction in zip(axes, ["D1 -> D2", "D2 -> D1"]):
        sub = df_main[df_main["direction"] == direction]
        x = np.arange(2)
        width = 0.19
        for s_idx, strat in enumerate(strategies_order):
            means, stds = [], []
            for arch in ["unet", "deeplabv3plus"]:
                vals = sub[(sub["arch"] == arch) & (sub["strategy"] == strat)]["cross_miou_fg"]
                means.append(float(vals.mean()))
                stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
            offset = (s_idx - 1.5) * width
            bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=strat, color=colors[strat])
            for bar, m_val, s_val in zip(bars, means, stds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    min(m_val + s_val + 0.02, 1.03),
                    f"{m_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(["U-Net (ResNet-34)", "DeepLabV3+ (ResNet-34)"])
        ax.set_title(f"Cross-Domain mIoU Foreground ({direction}, 3 Seed mean±std)")
        ax.set_ylabel("mIoU Foreground (0-1)")
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_ablation_overview_figure(per_member: Dict[str, pd.DataFrame], out_path: str) -> None:
    """Plot 4-panel ringkasan studi ablasi satu-faktor milik seluruh anggota tim."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    specs = [
        ("Attala", "Attala (230441100144): Strategi & Eval-Only", axes[0, 0]),
        ("Nafaul", "Nafaul (240441100125): Fungsi Loss", axes[0, 1]),
        ("M_Rafly", "M. Rafly (240441100086): Resolusi & Augmentasi", axes[1, 0]),
        ("Naufal", "Naufal (240441100038): Encoder & Fine-Tuning", axes[1, 1]),
    ]
    for key, title, ax in specs:
        sub = per_member.get(key, pd.DataFrame())
        if sub.empty:
            ax.set_title(f"{title} (Kosong)")
            ax.axis("off")
            continue
        labels = sub["strategy"].tolist()
        x = np.arange(len(labels))
        w = 0.36
        ax.bar(x - w / 2, sub["in_miou_fg"], w, label="In-Domain mIoU", color="#3B82F6")
        ax.bar(x + w / 2, sub["cross_miou_fg"], w, label="Cross-Domain mIoU", color="#F59E0B")
        for xi, v in zip(x + w / 2, sub["cross_miou_fg"]):
            ax.text(xi, float(v) + 0.015, f"{float(v):.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("mIoU Foreground")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
