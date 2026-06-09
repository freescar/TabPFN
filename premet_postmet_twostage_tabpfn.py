#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two-Stage TabPFN: PreMET Prediction → PostMET Prediction

Stage 1 (PreMET) — based on tabpfn_simple_probalistic.py:
  - Full slot/reference/temporal feature engineering
  - TabPFN probabilistic predictions with CI quantiles
  - Coverage-based metrics (ci_width thresholds + top-N% coverage selection)

Stage 2 (PostMET) — based on fit_loop_premet_postmet_tabpfn.py:
  - PostMET training data: ALL rows from --postmet-data-path are used to fit TabPFN
  - Inference performed on the *test split* of the PreMET dataset
  - Two parallel scenarios:
      A) pre_MET feature = actual PreMET (GroundTruth) from test rows
      B) pre_MET feature = Stage-1 predicted PreMET from test rows
  - Loop-count grouped metrics
  - Two PostMET visualisation figures

Outputs
-------
  <output_dir>/<dataset>_premet_timeseries.png   – Stage-1 prediction plot
  <output_dir>/<dataset>_postmet_timeseries.png  – PostMET time-series (both scenarios)
  <output_dir>/<dataset>_postmet_scatter.png     – PostMET scatter (both scenarios)
  <output_dir>/<dataset>_results.json            – All metrics (premet + loop-count + postmet)


  python premet_postmet_twostage_tabpfn.py \
  --premet-data-path /data/premet/ \
  --postmet-data-path /data/postmet_train.parquet \
  --output-dir ./results/twostage \
  --target-col GroundTruth \
  --postmet-pre-met-col BW092EH_MET \
  --postmet-post-met-col BW092WETEH_MET \
  --postmet-loop-count-col loop_count \
  --model-path /models/tabpfn-v3.ckpt
"""

import os

os.environ["TABPFN_NO_TELEMETRY"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["DISABLE_POSTHOG"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["SEGMENT_WRITE_KEY"] = ""
os.environ["ANALYTICS_DISABLED"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

import glob
import json
import time
import gc
import warnings
import argparse
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import LabelEncoder

from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Degrees of freedom")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore")


# ============================================================
# Defaults
# ============================================================

DEFAULT_PREMET_DATA_PATH = "/ossfs/workspace/tools/A2_DBJOA_BW09_Simple_Tabpfn_Tools/A2_DBJOA_BW09_Simple_Tabpfn_Tool01_CHA1.csv"
DEFAULT_POSTMET_DATA_PATH = "/ossfs/workspace/tools/A2_DBJOA_BW09_PLUS_20260101_20260601_merge_curr_pre_r2r_post_36tool.csv"          # required – path to a single postmet csv/parquet file

DEFAULT_OUTPUT_DIR = "./results/twostage"

# PreMET columns
DEFAULT_TARGET_COL = "GroundTruth"
DEFAULT_TIME_COL = "start_time"
DEFAULT_SLOT_COL = "slot_id"
DEFAULT_LOT_COL = "lot_id"
DEFAULT_WAFER_ID_COL = "wafer_id"
DEFAULT_REFERENCE_SLOT_IDS = "2,3,4,5,12,13,20,21,22,23"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.9

# PostMET columns (in both the postmet training file and in the premet dataset)
DEFAULT_POSTMET_PRE_MET_COL = "BW092EH_MET"
DEFAULT_POSTMET_POST_MET_COL = "BW092WETEH_MET"
DEFAULT_POSTMET_TOOL_COL = "tool_name"
DEFAULT_POSTMET_LOOP_COUNT_COL = "loop_count"
DEFAULT_POSTMET_SLOT_COL = "slot_id"

# Model
DEFAULT_MODEL_PATH = (
    "/ossfs/workspace/xrfm/TabPFN-main/models/"
    "tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt"
)
DEFAULT_N_ESTIMATORS = 4
DEFAULT_SOFTMAX_TEMPERATURE = 0.9
DEFAULT_AVERAGE_BEFORE_SOFTMAX = True
DEFAULT_POLY_FEATURES = 1
DEFAULT_SUBSAMPLE_SAMPLES = 2048
DEFAULT_PREDICT_BATCH_SIZE = 0

DEFAULT_MAX_FEATURES = 120
DEFAULT_MAX_MISSING_RATIO = 0.60
DEFAULT_MIN_VARIANCE = 1e-10

DEFAULT_TEMPORAL_LOT_WINDOW_K = 5
DEFAULT_RESIDUAL_PCA_COMPONENTS = 2
DEFAULT_LEARN_LOT_STATE = False
DEFAULT_LOT_STATE_DIMS = 2

DEFAULT_CI_QUANTILE_LOWER = 0.1
DEFAULT_CI_QUANTILE_UPPER = 0.9
DEFAULT_CONF_WIDTH_THRESHOLDS = "0.5,1.0,1.5"
DEFAULT_COVERAGE_THRESHOLDS = "0.10,0.20,0.30"

FB_DC_TARGET1 = 81.0
PRE_OFFSET = 0.3127
REC1_GRADIENT = 0.1313
LOOP_OFFSET = 6.0

RUN_VALUE_BOUNDS = np.array([0.0, 19.5, 26.2, 33.0, 39.8, 46.5, 53.5, 60.1, 100.0], dtype=np.float64)
CLASS_LABELS = np.arange(2, 10, dtype=int)


# ============================================================
# Loop 计算函数
# ============================================================
def ocd_to_run_value(ocd):
    """OCD → run_value 转换"""
    return (FB_DC_TARGET1 - np.asarray(ocd, dtype=np.float64) - PRE_OFFSET) / REC1_GRADIENT - LOOP_OFFSET


def run_value_to_loop(rv, out_of_range="clip"):
    """run_value → loop_count 转换"""
    rv = np.asarray(rv, dtype=np.float64)
    idx = np.searchsorted(RUN_VALUE_BOUNDS[1:-1], rv, side="right")
    loop = CLASS_LABELS[np.clip(idx, 0, len(CLASS_LABELS) - 1)].astype(float)
    if out_of_range == "nan":
        loop[~((rv >= RUN_VALUE_BOUNDS[0]) & (rv < RUN_VALUE_BOUNDS[-1]))] = np.nan
    return loop


def ocd_to_loop(ocd, out_of_range="clip"):
    """OCD → loop_count 直接转换"""
    return run_value_to_loop(ocd_to_run_value(ocd), out_of_range=out_of_range)

# ============================================================
# 从 Stage-1 预测计算 loop_count
# ============================================================
def add_loop_count_from_predictions(df_test, y_pred, target_col="GroundTruth"):
    """
    在 df_test 中添加 loop_count 列
    
    Parameters
    ----------
    df_test : pd.DataFrame
        Stage-1 test set (df_test from stage1_result)
    y_pred : np.ndarray
        Stage-1 预测值 (y_pred from stage1_result)
    target_col : str
        原始 OCD 列名
    
    Returns
    -------
    df_test : pd.DataFrame
        包含 loop_count 列的更新后的 DataFrame
    """
    df_test = df_test.copy()
    
    # 从预测值计算 loop_count
    predicted_loop = ocd_to_loop(y_pred, out_of_range="clip").astype(int)
    df_test["loop_count"] = predicted_loop
    
    print(f"  [INFO] 已从 Stage-1 预测值计算 loop_count")
    print(f"    预测 loop 分布: {dict(pd.Series(predicted_loop).value_counts().sort_index())}")
    
    return df_test

# ============================================================
# IO helpers
# ============================================================

def load_single_file(filepath: str) -> pd.DataFrame:
    if filepath.endswith(".parquet"):
        return pd.read_parquet(filepath)
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    raise ValueError(f"Unsupported file type: {filepath}")


def discover_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet"))) + sorted(
            glob.glob(os.path.join(path, "*.csv"))
        )
        if not files:
            raise FileNotFoundError(f"No parquet/csv in folder: {path}")
        return files
    raise FileNotFoundError(f"Path not found: {path}")


# ============================================================
# GPU cleanup
# ============================================================

def force_cleanup(light: bool = True) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if not light:
            torch.cuda.synchronize()


# ============================================================
# Metrics
# ============================================================

def acc_within(y_true: np.ndarray, y_pred: np.ndarray, thr: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= thr) * 100.0)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "acc05": float(acc_within(y_true, y_pred, 0.5)),
        "acc10": float(acc_within(y_true, y_pred, 1.0)),
    }


def eval_metrics_prob(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    conf_width_thresholds: list[float],
    n_total_test: Optional[int] = None,
) -> dict:
    """Probabilistic metrics including CI-width threshold subsets."""
    base = eval_metrics(y_true, y_pred)
    ci_width = q_upper - q_lower
    empirical_coverage = float(np.mean((y_true >= q_lower) & (y_true <= q_upper)) * 100.0)
    base.update({
        "ci_width_mean": float(np.mean(ci_width)),
        "ci_width_median": float(np.median(ci_width)),
        "ci_empirical_coverage_pct": empirical_coverage,
    })
    denominator = n_total_test if n_total_test is not None else len(y_true)
    for thr in conf_width_thresholds:
        high_conf = ci_width <= thr
        coverage_pct = float(np.mean(high_conf) * 100.0)
        coverage_pct_of_total = float(high_conf.sum() / denominator * 100.0)
        key_prefix = f"ci_thr{thr:.1f}"
        base[f"{key_prefix}_coverage_pct_of_total"] = coverage_pct_of_total
        base[f"{key_prefix}_coverage_pct_of_subset"] = coverage_pct
        if high_conf.sum() > 0:
            base[f"{key_prefix}_mae"] = float(
                mean_absolute_error(y_true[high_conf], y_pred[high_conf])
            )
            base[f"{key_prefix}_r2"] = float(r2_score(y_true[high_conf], y_pred[high_conf]))
            base[f"{key_prefix}_acc05"] = float(
                acc_within(y_true[high_conf], y_pred[high_conf], 0.5)
            )
            base[f"{key_prefix}_acc10"] = float(
                acc_within(y_true[high_conf], y_pred[high_conf], 1.0)
            )
        else:
            base[f"{key_prefix}_mae"] = float("nan")
            base[f"{key_prefix}_r2"] = float("nan")
            base[f"{key_prefix}_acc05"] = float("nan")
            base[f"{key_prefix}_acc10"] = float("nan")
    return base


def eval_metrics_by_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    coverage_thresholds: list[float],
    n_total_test: Optional[int] = None,
) -> dict:
    """Metrics on the top-N% narrowest-CI samples."""
    result = {}
    n_subset = len(y_true)
    n_total = n_total_test if n_total_test is not None else n_subset
    ci_width = q_upper - q_lower
    sorted_idx = np.argsort(ci_width)
    for cov_pct in coverage_thresholds:
        n_select = max(1, int(n_subset * cov_pct))
        selected_idx = sorted_idx[:n_select]
        y_true_sel = y_true[selected_idx]
        y_pred_sel = y_pred[selected_idx]
        q_lower_sel = q_lower[selected_idx]
        q_upper_sel = q_upper[selected_idx]
        metrics_sel = eval_metrics(y_true_sel, y_pred_sel)
        ci_width_sel = q_upper_sel - q_lower_sel
        empirical_coverage_sel = float(
            np.mean((y_true_sel >= q_lower_sel) & (y_true_sel <= q_upper_sel)) * 100.0
        )
        key = f"cov_{cov_pct * 100:.0f}pct"
        result[key] = {
            "n_samples": int(n_select),
            "coverage_pct_of_subset": float(n_select / n_subset * 100.0),
            "coverage_pct_of_total": float(n_select / n_total * 100.0),
            "mae": metrics_sel["mae"],
            "r2": metrics_sel["r2"],
            "acc05": metrics_sel["acc05"],
            "acc10": metrics_sel["acc10"],
            "ci_width_mean": float(np.mean(ci_width_sel)),
            "ci_empirical_coverage_pct": empirical_coverage_sel,
        }
    return result


def eval_metrics_by_loop_count(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loop_counts: np.ndarray,
) -> dict:
    """Per-loop-count metrics."""
    result = {}
    for lc in np.unique(loop_counts):
        mask = loop_counts == lc
        if mask.sum() == 0:
            continue
        m = eval_metrics(y_true[mask], y_pred[mask])
        m["n_samples"] = int(mask.sum())
        result[f"loop_{lc}"] = m
    return result


# ============================================================
# Plots
# ============================================================

def plot_pred_true_timeseries(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_is_ref: np.ndarray,
    title: str,
    out_path: str,
    ylabel: str,
    q_lower: Optional[np.ndarray] = None,
    q_upper: Optional[np.ndarray] = None,
) -> None:
    """Stage-1 PreMET time-series plot."""
    x = np.arange(len(y_test))
    is_nonref = ~test_is_ref

    plt.figure(figsize=(18, 6))
    plt.fill_between(x, y_test - 0.5, y_test + 0.5, alpha=0.10, color="green", label="±0.5 band")
    if q_lower is not None and q_upper is not None:
        plt.fill_between(
            x[is_nonref], q_lower[is_nonref], q_upper[is_nonref],
            alpha=0.18, color="steelblue", label="pred CI (non-ref)",
        )
    plt.plot(x, y_test, color="black", alpha=0.35, linewidth=1.0, label="true (all)")
    plt.scatter(x[is_nonref], y_test[is_nonref], s=8, color="black", alpha=0.6, label="true (non-ref)")
    plt.scatter(x[test_is_ref], y_test[test_is_ref], s=8, color="gray", alpha=0.4, label="true (ref)")
    plt.plot(x, y_pred, color="steelblue", alpha=0.55, linewidth=1.2, label="pred")
    plt.scatter(x[is_nonref], y_pred[is_nonref], s=8, color="steelblue", alpha=0.6, label="pred (non-ref)")
    plt.scatter(x[test_is_ref], y_pred[test_is_ref], s=8, color="salmon", alpha=0.4, label="pred (ref)")
    plt.title(title)
    plt.xlabel("test sample index (time order)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_postmet_timeseries(
    y_post_from_gt: np.ndarray,
    y_post_from_pred: np.ndarray,
    metrics: dict,
    out_path: str,
    dataset_name: str,
) -> None:
    """PostMET time-series comparing the two predicted curves.

    Scenario A — post_MET predicted from GROUNDTRUTH PreMET (reference)
    Scenario B — post_MET predicted from Stage-1 PREDICTED  PreMET (deployed)
    """
    x = np.arange(len(y_post_from_gt))
    diff = y_post_from_pred - y_post_from_gt

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(
        f"PostMET Prediction — {dataset_name}\n"
        f"Predicted-PreMET vs Groundtruth-PreMET pipeline:  "
        f"MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}  "
        f"Acc@0.5={metrics['acc05']:.1f}%  Acc@1.0={metrics['acc10']:.1f}%",
        fontsize=11,
    )

    ax = axes[0]
    ax.plot(x, y_post_from_gt, color="black", linewidth=1.2, alpha=0.7,
            label="post_MET (Scenario A — groundtruth PreMET)")
    ax.plot(x, y_post_from_pred, color="tomato", linewidth=1.2, alpha=0.85, linestyle="--",
            label="post_MET (Scenario B — predicted PreMET)")
    ax.fill_between(x, y_post_from_gt - 0.5, y_post_from_gt + 0.5, alpha=0.08, color="green",
                    label="±0.5 band around Scenario A")
    ax.set_ylabel("post_MET")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_title("Predicted post_MET: groundtruth-PreMET (reference) vs predicted-PreMET")

    ax = axes[1]
    ax.plot(x, diff, color="steelblue", linewidth=1.0, alpha=0.8,
            label="Scenario B − Scenario A")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.fill_between(x, -0.5, 0.5, alpha=0.08, color="green", label="±0.5 band")
    ax.set_xlabel("test sample index (time order)")
    ax.set_ylabel("Δ post_MET")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_title("Difference between the two predicted post_MET curves")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_postmet_scatter(
    y_post_from_gt: np.ndarray,
    y_post_from_pred: np.ndarray,
    metrics: dict,
    loop_counts: Optional[np.ndarray],
    loop_metrics: dict,
    out_path: str,
    dataset_name: str,
) -> None:
    """PostMET scatter + difference distribution + per-loop-count metrics,
    comparing the two predicted post_MET curves (predicted-PreMET vs
    groundtruth-PreMET)."""
    res = y_post_from_pred - y_post_from_gt
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"PostMET Analysis — {dataset_name}", fontsize=13, fontweight="bold")

    # Scatter: Scenario A (x) vs Scenario B (y)
    ax = axes[0, 0]
    ax.scatter(y_post_from_gt, y_post_from_pred, s=8, alpha=0.5, color="steelblue")
    lo = float(min(y_post_from_gt.min(), y_post_from_pred.min()))
    hi = float(max(y_post_from_gt.max(), y_post_from_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    ax.set_xlabel("post_MET (Scenario A — groundtruth PreMET)")
    ax.set_ylabel("post_MET (Scenario B — predicted PreMET)")
    ax.set_title(
        f"Predicted-PreMET vs Groundtruth-PreMET\n"
        f"MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}",
        fontsize=10,
    )
    ax.grid(alpha=0.25)

    # Difference distribution
    ax = axes[0, 1]
    ax.hist(res, bins=30, alpha=0.7, color="tomato", label=f"mean={res.mean():.3f}")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Δ post_MET (Scenario B − Scenario A)")
    ax.set_ylabel("Count")
    ax.set_title("Difference Distribution", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    # Absolute difference over samples
    ax = axes[1, 0]
    x = np.arange(len(y_post_from_gt))
    ax.plot(x, np.abs(res), color="steelblue", linewidth=0.8, alpha=0.7, label="|Δ post_MET|")
    ax.axhline(metrics["mae"], color="tomato", linestyle="--", linewidth=1.2,
               label=f"MAE={metrics['mae']:.4f}")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Absolute Difference")
    ax.set_title("Absolute Difference over Samples", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Per-loop-count MAE (grouped by groundtruth loop)
    ax = axes[1, 1]
    if loop_metrics:
        lc_keys = sorted(loop_metrics.keys(), key=lambda k: int(k.split("_")[1]))
        lc_labels = [k.replace("loop_", "lc=") for k in lc_keys]
        mae_vals = [loop_metrics[k]["mae"] for k in lc_keys]
        xpos = np.arange(len(lc_keys))
        ax.bar(xpos, mae_vals, 0.6, color="steelblue", alpha=0.85)
        ax.set_xticks(xpos)
        ax.set_xticklabels(lc_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("MAE (B vs A)")
        ax.set_title("Per-Loop-Count MAE (grouped by groundtruth loop)", fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "loop_count metrics\nnot available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# ============================================================
# Slot / reference-MET feature engineering  (Stage 1)
# ============================================================

def build_slot_ref_features(
    df: pd.DataFrame,
    *,
    target_col: str,
    slot_col: str,
    lot_col: str,
    reference_slot_ids: list[int],
) -> pd.DataFrame:
    slots = df[slot_col].to_numpy(dtype=np.int32)
    lots = df[lot_col].to_numpy()
    mets = df[target_col].to_numpy(dtype=np.float32)
    n_rows = len(df)

    is_ref = np.isin(slots, reference_slot_ids)
    slot_vals = df[slot_col].values
    slot_min = float(slot_vals.min())
    slot_max = float(slot_vals.max())
    slot_range = max(slot_max - slot_min, 1.0)
    slot_norm = (slots - slot_min) / slot_range
    slot_center_dist = np.abs(slot_norm - 0.5)

    feat: dict[str, np.ndarray] = {
        "slot_id": slots.astype(np.float32),
        "slot_norm": slot_norm.astype(np.float32),
        "slot_center_dist": slot_center_dist.astype(np.float32),
        "slot_trend_sq": (slot_norm ** 2).astype(np.float32),
        "slot_trend_cubic": (slot_norm ** 3).astype(np.float32),
        "is_ref_slot": is_ref.astype(np.float32),
    }

    ref_ids_arr = np.array(sorted(set(reference_slot_ids)), dtype=float)
    if len(ref_ids_arr) > 0:
        nearest_ref_dist = np.min(
            np.abs(slots[:, None].astype(float) - ref_ids_arr[None, :]), axis=1
        ).astype(np.float32)
    else:
        nearest_ref_dist = np.zeros(n_rows, dtype=np.float32)
    feat["nearest_ref_dist"] = nearest_ref_dist

    lot_ref_met_mean = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_std = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_median = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_min = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_max = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_range = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_count = np.zeros(n_rows, dtype=np.float32)
    ref_slot_mets: dict[int, np.ndarray] = {
        sid: np.full(n_rows, np.nan, dtype=np.float32) for sid in reference_slot_ids
    }
    ref_met_interp = np.full(n_rows, np.nan, dtype=np.float32)

    for lot in np.unique(lots):
        lot_mask = lots == lot
        lot_ref_mask = lot_mask & is_ref
        n_ref = int(lot_ref_mask.sum())
        if n_ref == 0:
            continue
        lot_ref_slots = slots[lot_ref_mask]
        lot_ref_mets = mets[lot_ref_mask]
        slot_met_dict: dict[int, float] = {}
        for s, m in zip(lot_ref_slots.tolist(), lot_ref_mets.tolist()):
            slot_met_dict[s] = m
        for sid in reference_slot_ids:
            if sid in slot_met_dict:
                ref_slot_mets[sid][lot_mask] = slot_met_dict[sid]
        total_sum = float(np.nansum(lot_ref_mets))
        total_sum2 = float(np.nansum(lot_ref_mets ** 2))
        sort_order = np.argsort(lot_ref_slots)
        sorted_ref_slots = lot_ref_slots[sort_order].astype(float)
        sorted_ref_mets = lot_ref_mets[sort_order]
        nonref_indices = np.where(lot_mask & ~is_ref)[0]
        if len(nonref_indices) > 0:
            mean_val = total_sum / n_ref
            var_val = max(total_sum2 / n_ref - mean_val ** 2, 0.0)
            lot_ref_met_mean[nonref_indices] = mean_val
            lot_ref_met_std[nonref_indices] = float(np.sqrt(var_val))
            lot_ref_met_median[nonref_indices] = float(np.nanmedian(lot_ref_mets))
            lot_ref_met_min[nonref_indices] = float(np.nanmin(lot_ref_mets))
            lot_ref_met_max[nonref_indices] = float(np.nanmax(lot_ref_mets))
            lot_ref_met_range[nonref_indices] = float(
                np.nanmax(lot_ref_mets) - np.nanmin(lot_ref_mets)
            )
            lot_ref_met_count[nonref_indices] = float(n_ref)
            for idx in nonref_indices:
                ref_met_interp[idx] = float(
                    np.interp(slots[idx], sorted_ref_slots, sorted_ref_mets)
                )
        ref_indices = np.where(lot_ref_mask)[0]
        for idx in ref_indices:
            curr_slot = int(slots[idx])
            own_met = slot_met_dict.get(curr_slot)
            if own_met is None:
                continue
            n_loo = n_ref - 1
            if n_loo == 0:
                continue
            loo_sum = total_sum - own_met
            loo_mean = loo_sum / n_loo
            loo_sum2 = total_sum2 - own_met ** 2
            loo_std = float(np.sqrt(max(loo_sum2 / n_loo - loo_mean ** 2, 0.0)))
            loo_mets = np.array(
                [m for s, m in slot_met_dict.items() if s != curr_slot], dtype=np.float32
            )
            lot_ref_met_mean[idx] = loo_mean
            lot_ref_met_std[idx] = loo_std
            lot_ref_met_median[idx] = float(np.nanmedian(loo_mets))
            lot_ref_met_min[idx] = float(np.nanmin(loo_mets))
            lot_ref_met_max[idx] = float(np.nanmax(loo_mets))
            lot_ref_met_range[idx] = float(np.nanmax(loo_mets) - np.nanmin(loo_mets))
            lot_ref_met_count[idx] = float(n_loo)
            loo_items = [(s, slot_met_dict[s]) for s in slot_met_dict if s != curr_slot]
            loo_slots = np.array([s for s, _ in loo_items], dtype=float)
            loo_met_vals = np.array([m for _, m in loo_items], dtype=np.float32)
            if len(loo_slots) >= 1:
                order = np.argsort(loo_slots)
                ref_met_interp[idx] = float(
                    np.interp(float(curr_slot), loo_slots[order], loo_met_vals[order])
                )
            ref_slot_mets[curr_slot][idx] = np.nan

    feat.update({
        "lot_ref_met_mean": lot_ref_met_mean,
        "lot_ref_met_std": lot_ref_met_std,
        "lot_ref_met_median": lot_ref_met_median,
        "lot_ref_met_min": lot_ref_met_min,
        "lot_ref_met_max": lot_ref_met_max,
        "lot_ref_met_range": lot_ref_met_range,
        "lot_ref_met_count": lot_ref_met_count,
        "ref_met_interp": ref_met_interp,
    })
    for sid in reference_slot_ids:
        feat[f"ref_slot_{sid}_met"] = ref_slot_mets[sid]
        feat[f"ref_slot_{sid}_met_dev"] = ref_slot_mets[sid] - lot_ref_met_mean

    result = pd.DataFrame(feat, index=df.index)
    for col in result.columns:
        if result[col].isna().any():
            col_median = result[col].median()
            if np.isnan(col_median):
                col_median = 0.0
            result[col] = result[col].fillna(col_median)
    return result


def build_temporal_lot_features(
    df: pd.DataFrame,
    *,
    time_col: str,
    lot_col: str,
    target_col: str,
    slot_col: str,
    reference_slot_ids: list[int],
    window_k: int = 5,
) -> pd.DataFrame:
    lots = df[lot_col].to_numpy()
    mets = df[target_col].to_numpy(dtype=np.float32)
    slots = df[slot_col].to_numpy()
    times = df[time_col].to_numpy()
    n_rows = len(df)
    is_ref = np.isin(slots, reference_slot_ids)

    unique_lots = np.unique(lots)
    lot_earliest_time: dict = {}
    lot_ref_mean: dict = {}

    for lot in unique_lots:
        lot_mask = lots == lot
        lot_times = times[lot_mask]
        try:
            lot_earliest_time[lot] = np.min(lot_times)
        except Exception:
            lot_earliest_time[lot] = lot_times[0]
        lot_ref_mask = lot_mask & is_ref
        if lot_ref_mask.any():
            lot_ref_mean[lot] = float(np.nanmean(mets[lot_ref_mask]))
        else:
            lot_ref_mean[lot] = np.nan

    try:
        lots_sorted = sorted(unique_lots, key=lambda lo: lot_earliest_time[lo])
    except Exception:
        lots_sorted = list(unique_lots)

    lot_rank = {lot: i for i, lot in enumerate(lots_sorted)}

    prev_k_ref_mean = np.full(n_rows, np.nan, dtype=np.float32)
    prev_k_ref_std = np.full(n_rows, np.nan, dtype=np.float32)
    prev_k_ref_min = np.full(n_rows, np.nan, dtype=np.float32)
    prev_k_ref_max = np.full(n_rows, np.nan, dtype=np.float32)
    prev_k_ref_trend = np.full(n_rows, np.nan, dtype=np.float32)
    prev_lot_ref_mean_arr = np.full(n_rows, np.nan, dtype=np.float32)
    lot_rank_arr = np.full(n_rows, np.nan, dtype=np.float32)
    lot_time_gap_arr = np.full(n_rows, np.nan, dtype=np.float32)

    for lot in unique_lots:
        lot_mask = lots == lot
        rank = lot_rank[lot]
        lot_rank_arr[lot_mask] = float(rank)
        preceding = lots_sorted[max(0, rank - window_k): rank]
        if not preceding:
            continue
        prev_means = [
            lot_ref_mean[pl]
            for pl in preceding
            if not np.isnan(lot_ref_mean.get(pl, np.nan))
        ]
        if not prev_means:
            continue
        prev_arr = np.array(prev_means, dtype=np.float32)
        prev_k_ref_mean[lot_mask] = float(np.nanmean(prev_arr))
        prev_k_ref_std[lot_mask] = float(np.nanstd(prev_arr)) if len(prev_arr) > 1 else 0.0
        prev_k_ref_min[lot_mask] = float(np.nanmin(prev_arr))
        prev_k_ref_max[lot_mask] = float(np.nanmax(prev_arr))
        prev_lot_ref_mean_arr[lot_mask] = float(lot_ref_mean.get(preceding[-1], np.nan))
        if len(prev_arr) >= 2:
            x_t = np.arange(len(prev_arr), dtype=np.float32)
            x_c = x_t - x_t.mean()
            denom = float(np.dot(x_c, x_c))
            slope = (
                float(np.dot(x_c, prev_arr - float(prev_arr.mean())) / denom)
                if denom > 0.0
                else 0.0
            )
            prev_k_ref_trend[lot_mask] = slope
        else:
            prev_k_ref_trend[lot_mask] = 0.0
        prev_lot = preceding[-1]
        try:
            curr_t = lot_earliest_time[lot]
            prev_t = lot_earliest_time[prev_lot]
            delta = curr_t - prev_t
            if hasattr(delta, "total_seconds"):
                gap_h = float(delta.total_seconds()) / 3600.0
            elif hasattr(delta, "astype"):
                gap_h = float(delta.astype("timedelta64[s]").astype(np.float64)) / 3600.0
            else:
                gap_h = float(delta) / 3600.0
            lot_time_gap_arr[lot_mask] = gap_h
        except Exception:
            pass

    result = pd.DataFrame(
        {
            "prev_k_lots_ref_mean": prev_k_ref_mean,
            "prev_k_lots_ref_std": prev_k_ref_std,
            "prev_k_lots_ref_min": prev_k_ref_min,
            "prev_k_lots_ref_max": prev_k_ref_max,
            "prev_k_lots_ref_trend": prev_k_ref_trend,
            "prev_lot_ref_mean": prev_lot_ref_mean_arr,
            "lot_rank_in_window": lot_rank_arr,
            "lot_time_gap_hours": lot_time_gap_arr,
        },
        index=df.index,
    )
    for col in result.columns:
        if result[col].isna().any():
            med = result[col].median()
            if np.isnan(med):
                med = 0.0
            result[col] = result[col].fillna(med)
    return result


def _build_lot_reference_profiles(
    df: pd.DataFrame,
    *,
    target_col: str,
    slot_col: str,
    lot_col: str,
    reference_slot_ids: list[int],
) -> dict:
    slots = df[slot_col].to_numpy()
    lots = df[lot_col].to_numpy()
    mets = df[target_col].to_numpy(dtype=np.float32)
    is_ref = np.isin(slots, reference_slot_ids)
    ref_ids = list(reference_slot_ids)
    lot_profiles: dict = {}
    for lot in np.unique(lots):
        lot_mask = lots == lot
        lot_ref_mask = lot_mask & is_ref
        profile = np.full(len(ref_ids), np.nan, dtype=np.float32)
        if lot_ref_mask.any():
            lot_ref_slots = slots[lot_ref_mask].astype(int)
            lot_ref_mets = mets[lot_ref_mask]
            for j, sid in enumerate(ref_ids):
                sid_vals = lot_ref_mets[lot_ref_slots == sid]
                if len(sid_vals) > 0:
                    profile[j] = float(np.nanmean(sid_vals))
        lot_profiles[lot] = profile
    return lot_profiles


def fit_global_reference_model(
    df_train: pd.DataFrame,
    *,
    target_col: str,
    slot_col: str,
    lot_col: str,
    reference_slot_ids: list[int],
    n_components: int = 3,
    n_residual_components: int = 2,
) -> dict:
    ref_ids = list(reference_slot_ids)
    n_ref = len(ref_ids)
    if n_ref == 0:
        return {
            "reference_slot_ids": np.array([], dtype=np.int32),
            "slot_fill_values": np.array([], dtype=np.float32),
            "template_profile": np.array([], dtype=np.float32),
            "components": np.empty((0, 0), dtype=np.float32),
            "residual_components": np.empty((0, 0), dtype=np.float32),
        }
    lot_profiles = _build_lot_reference_profiles(
        df_train,
        target_col=target_col,
        slot_col=slot_col,
        lot_col=lot_col,
        reference_slot_ids=ref_ids,
    )
    mat = (
        np.vstack(list(lot_profiles.values())).astype(np.float32)
        if lot_profiles
        else np.empty((0, n_ref), dtype=np.float32)
    )
    if mat.shape[0] == 0:
        slot_fill_values = np.zeros(n_ref, dtype=np.float32)
        template_profile = np.zeros(n_ref, dtype=np.float32)
        components = np.empty((0, n_ref), dtype=np.float32)
        residual_components = np.empty((0, n_ref), dtype=np.float32)
    else:
        slot_fill_values = np.zeros(n_ref, dtype=np.float32)
        for j in range(n_ref):
            col = mat[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                slot_fill_values[j] = float(np.median(valid))
        mat_filled = np.where(np.isnan(mat), slot_fill_values[None, :], mat).astype(np.float32)
        template_profile = np.mean(mat_filled, axis=0).astype(np.float32)
        centered = mat_filled - template_profile[None, :]
        n_comp = min(int(n_components), centered.shape[0], centered.shape[1])
        if n_comp <= 0:
            components = np.empty((0, n_ref), dtype=np.float32)
            residual_components = np.empty((0, n_ref), dtype=np.float32)
        else:
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                components = vh[:n_comp].astype(np.float32)
            except np.linalg.LinAlgError:
                components = np.empty((0, n_ref), dtype=np.float32)
            n_resid_comp = min(int(n_residual_components), centered.shape[0], centered.shape[1])
            if n_resid_comp <= 0 or components.shape[0] == 0:
                residual_components = np.empty((0, n_ref), dtype=np.float32)
            else:
                try:
                    proj = centered @ components.T
                    reconstruction = proj @ components
                    resid_mat = centered - reconstruction
                    _, _, vh_r = np.linalg.svd(resid_mat, full_matrices=False)
                    residual_components = vh_r[:n_resid_comp].astype(np.float32)
                except np.linalg.LinAlgError:
                    residual_components = np.empty((0, n_ref), dtype=np.float32)
    return {
        "reference_slot_ids": np.asarray(ref_ids, dtype=np.int32),
        "slot_fill_values": slot_fill_values,
        "template_profile": template_profile,
        "components": components,
        "residual_components": residual_components,
    }


def append_global_reference_features(
    df: pd.DataFrame,
    X_base: pd.DataFrame,
    *,
    target_col: str,
    slot_col: str,
    lot_col: str,
    reference_slot_ids: list[int],
    global_ref_model: Optional[dict],
) -> pd.DataFrame:
    if global_ref_model is None or len(reference_slot_ids) == 0:
        return X_base
    template_profile = np.asarray(
        global_ref_model.get("template_profile", np.array([])), dtype=np.float32
    )
    slot_fill_values = np.asarray(
        global_ref_model.get("slot_fill_values", np.array([])), dtype=np.float32
    )
    components = np.asarray(
        global_ref_model.get("components", np.empty((0, 0))), dtype=np.float32
    )
    residual_components = np.asarray(
        global_ref_model.get("residual_components", np.empty((0, 0))), dtype=np.float32
    )
    model_ref_ids = np.asarray(
        global_ref_model.get(
            "reference_slot_ids", np.asarray(reference_slot_ids, dtype=np.int32)
        ),
        dtype=np.int32,
    )
    if (
        len(template_profile) == 0
        or len(template_profile) != len(reference_slot_ids)
        or len(slot_fill_values) != len(reference_slot_ids)
    ):
        return X_base

    X = X_base.copy()
    slots = df[slot_col].to_numpy(dtype=np.int32)
    lots = df[lot_col].to_numpy()
    lot_profiles = _build_lot_reference_profiles(
        df,
        target_col=target_col,
        slot_col=slot_col,
        lot_col=lot_col,
        reference_slot_ids=reference_slot_ids,
    )
    n_rows = len(df)
    n_comp = int(components.shape[0]) if components.ndim == 2 else 0
    n_resid_comp = int(residual_components.shape[0]) if residual_components.ndim == 2 else 0
    eps = 1e-8
    template_norm = float(np.linalg.norm(template_profile))
    template_norm2 = float(np.dot(template_profile, template_profile))
    template_centered = template_profile - float(np.mean(template_profile))
    template_centered_norm = float(np.linalg.norm(template_centered))
    interp_ref_ids = model_ref_ids.astype(np.float32)
    interp_template_profile = template_profile.copy()
    if len(interp_ref_ids) >= 2:
        sort_idx = np.argsort(interp_ref_ids)
        interp_ref_ids = interp_ref_ids[sort_idx]
        interp_template_profile = interp_template_profile[sort_idx]
    slot_min = float(slots.min())
    slot_range = max(float(slots.max()) - slot_min, 1.0)
    slot_norm_all = (slots.astype(np.float32) - slot_min) / slot_range

    global_ref_profile_mean = np.full(n_rows, np.nan, dtype=np.float32)
    global_ref_profile_std = np.full(n_rows, np.nan, dtype=np.float32)
    global_ref_template_cos = np.full(n_rows, np.nan, dtype=np.float32)
    global_ref_template_proj = np.full(n_rows, np.nan, dtype=np.float32)
    global_ref_profile_corr = np.full(n_rows, np.nan, dtype=np.float32)
    global_ref_resid_rmse = np.full(n_rows, np.nan, dtype=np.float32)
    global_ref_template_interp = np.full(n_rows, np.nan, dtype=np.float32)
    pc_scores = np.zeros((n_rows, n_comp), dtype=np.float32)
    resid_pc_scores = np.zeros((n_rows, n_resid_comp), dtype=np.float32)

    for lot in np.unique(lots):
        lot_mask = lots == lot
        lot_idx = np.where(lot_mask)[0]
        raw_profile = lot_profiles.get(
            lot, np.full(len(reference_slot_ids), np.nan, dtype=np.float32)
        )
        filled_profile = np.where(np.isnan(raw_profile), slot_fill_values, raw_profile).astype(
            np.float32
        )
        centered_profile = filled_profile - template_profile
        profile_norm = float(np.linalg.norm(filled_profile))
        profile_centered = filled_profile - float(np.mean(filled_profile))
        profile_centered_norm = float(np.linalg.norm(profile_centered))
        cos_val = float(
            np.dot(filled_profile, template_profile) / (profile_norm * template_norm + eps)
        )
        proj_val = float(np.dot(filled_profile, template_profile) / (template_norm2 + eps))
        corr_val = float(
            np.dot(profile_centered, template_centered)
            / (profile_centered_norm * template_centered_norm + eps)
        )
        rmse_val = float(np.sqrt(np.mean(centered_profile ** 2)))
        global_ref_profile_mean[lot_idx] = float(np.mean(filled_profile))
        global_ref_profile_std[lot_idx] = float(np.std(filled_profile))
        global_ref_template_cos[lot_idx] = cos_val
        global_ref_template_proj[lot_idx] = proj_val
        global_ref_profile_corr[lot_idx] = corr_val
        global_ref_resid_rmse[lot_idx] = rmse_val
        if n_comp > 0:
            lot_scores = centered_profile @ components.T
            pc_scores[lot_idx, :] = lot_scores[None, :]
        if n_resid_comp > 0 and residual_components.shape[1] == len(reference_slot_ids):
            lot_resid_vec = (
                centered_profile - (centered_profile @ components.T) @ components
                if n_comp > 0
                else centered_profile
            )
            resid_pc_scores[lot_idx, :] = (lot_resid_vec @ residual_components.T)[None, :]
        if len(interp_ref_ids) >= 2:
            global_ref_template_interp[lot_idx] = np.interp(
                slots[lot_idx], interp_ref_ids, interp_template_profile
            )
        elif len(interp_ref_ids) == 1:
            global_ref_template_interp[lot_idx] = interp_template_profile[0]
        else:
            global_ref_template_interp[lot_idx] = 0.0

    X["global_ref_profile_mean"] = global_ref_profile_mean
    X["global_ref_profile_std"] = global_ref_profile_std
    X["global_ref_template_cos"] = global_ref_template_cos
    X["global_ref_template_proj"] = global_ref_template_proj
    X["global_ref_profile_corr"] = global_ref_profile_corr
    X["global_ref_resid_rmse"] = global_ref_resid_rmse
    X["global_ref_template_interp"] = global_ref_template_interp
    if "ref_met_interp" in X.columns:
        X["global_ref_interp_resid"] = (
            X["ref_met_interp"].to_numpy(dtype=np.float32) - global_ref_template_interp
        )
    for j in range(n_comp):
        X[f"global_ref_pc{j + 1}"] = pc_scores[:, j]
    for j in range(n_resid_comp):
        col_name = f"global_ref_resid_pc{j + 1}"
        X[col_name] = resid_pc_scores[:, j]
        X[f"{col_name}_x_slot"] = resid_pc_scores[:, j] * slot_norm_all
    return X


def _coerce_mixed_columns_for_tabpfn(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        dt = X[c].dtype
        if pd.api.types.is_object_dtype(dt):
            nunique = X[c].nunique(dropna=True)
            if nunique <= max(100, int(len(X) * 0.2)):
                X[c] = X[c].astype("category")
        elif pd.api.types.is_numeric_dtype(dt):
            X[c] = X[c].astype(np.float32)
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
    return X


def fast_select_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_all: pd.DataFrame,
    *,
    max_features: int,
    max_missing_ratio: float,
    min_variance: float,
) -> tuple[pd.DataFrame, list[str], dict]:
    info = {
        "raw_features": int(X_all.shape[1]),
        "after_missing_filter": 0,
        "after_variance_filter": 0,
        "after_score_filter": 0,
    }
    cols = list(X_train.columns)
    if not cols:
        return X_all, cols, info
    miss_ratio = X_train.isna().mean()
    keep_cols = miss_ratio[miss_ratio <= max_missing_ratio].index.tolist()
    if not keep_cols:
        keep_cols = cols
    X_train_1 = X_train[keep_cols]
    info["after_missing_filter"] = int(len(keep_cols))
    num_cols = X_train_1.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        selected = keep_cols[:max_features]
        info["after_variance_filter"] = int(len(selected))
        info["after_score_filter"] = int(len(selected))
        return X_all[selected], selected, info
    X_num = X_train_1[num_cols]
    variances = X_num.var(axis=0, skipna=True)
    keep_num_cols = variances[variances > min_variance].index.tolist()
    if not keep_num_cols:
        keep_num_cols = num_cols
    X_num = X_num[keep_num_cols]
    info["after_variance_filter"] = int(len(keep_num_cols))
    X_fill = X_num.fillna(X_num.median(numeric_only=True))
    try:
        scores, _ = f_regression(X_fill, y_train)
        scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
        order = np.argsort(scores)[::-1]
        top_idx = order[: min(max_features, len(keep_num_cols))]
        selected = [keep_num_cols[i] for i in top_idx]
    except Exception:
        selected = keep_num_cols[:max_features]
    info["after_score_filter"] = int(len(selected))
    return X_all[selected], selected, info


# ============================================================
# TabPFN model helpers
# ============================================================

def create_model(
    model_path: str,
    n_estimators: int,
    softmax_temperature: float,
    average_before_softmax: bool,
    poly_features: int,
    subsample_samples: int,
) -> TabPFNRegressor:
    return TabPFNRegressor(
        model_path=model_path,
        device="cuda",
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        average_before_softmax=average_before_softmax,
        memory_saving_mode=True,
        ignore_pretraining_limits=True,
        inference_config={
            "SUBSAMPLE_SAMPLES": max(256, int(subsample_samples)),
            "POLYNOMIAL_FEATURES": max(1, int(poly_features)),
        },
    )


def predict_maybe_batched_with_quantiles(
    model: TabPFNRegressor,
    X: pd.DataFrame,
    batch_size: int,
    ci_quantiles: list[float],
) -> tuple[np.ndarray, list[np.ndarray]]:
    def _predict_batch(batch: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
        result = model.predict(batch, output_type="main", quantiles=ci_quantiles)
        return result["mean"], result["quantiles"]

    if batch_size is None or batch_size <= 0 or len(X) <= batch_size:
        return _predict_batch(X)
    means: list[np.ndarray] = []
    quantile_parts: list[list[np.ndarray]] = [[] for _ in ci_quantiles]
    for i in range(0, len(X), batch_size):
        batch_mean, batch_quantiles = _predict_batch(X.iloc[i: i + batch_size])
        means.append(batch_mean)
        for j, q_arr in enumerate(batch_quantiles):
            quantile_parts[j].append(q_arr)
    return np.concatenate(means), [np.concatenate(parts) for parts in quantile_parts]


def apply_residual_compensation(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lot_col: str,
    slot_col: str,
    reference_slot_ids: list[int],
) -> np.ndarray:
    compensated = y_pred.copy()
    lots = df_meta[lot_col].values
    slots = df_meta[slot_col].values
    is_ref = np.isin(slots, reference_slot_ids)
    for lot in np.unique(lots):
        lot_mask = lots == lot
        lot_ref_mask = lot_mask & is_ref
        if lot_ref_mask.sum() == 0:
            continue
        bias = np.nanmean(y_true[lot_ref_mask] - y_pred[lot_ref_mask])
        if np.isnan(bias):
            continue
        compensated[lot_mask & ~is_ref] += bias
    return compensated


def fit_lot_latent_states(
    model: TabPFNRegressor,
    X_selected: pd.DataFrame,
    y: np.ndarray,
    df_meta: pd.DataFrame,
    *,
    lot_col: str,
    slot_col: str,
    reference_slot_ids: list[int],
    n_dims: int = 2,
) -> pd.DataFrame:
    n_dims = max(1, int(n_dims))
    lots = df_meta[lot_col].to_numpy()
    slots = df_meta[slot_col].to_numpy(dtype=np.float32)
    is_ref = np.isin(slots, list(reference_slot_ids))
    n_rows = len(df_meta)
    preds = model.predict(X_selected)
    slot_min = float(slots.min())
    slot_range = max(float(slots.max()) - slot_min, 1.0)
    slot_norm = (slots - slot_min) / slot_range
    latent = np.zeros((n_rows, n_dims), dtype=np.float32)
    for lot in np.unique(lots):
        lot_mask = lots == lot
        lot_ref_mask = lot_mask & is_ref
        n_ref_in_lot = int(lot_ref_mask.sum())
        if n_ref_in_lot == 0:
            continue
        resid = (y[lot_ref_mask] - preds[lot_ref_mask]).astype(np.float64)
        slot_norm_ref = slot_norm[lot_ref_mask].astype(np.float64)
        n_basis = min(n_dims, n_ref_in_lot)
        phi_ref = np.column_stack([slot_norm_ref ** d for d in range(n_basis)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(phi_ref, resid, rcond=None)
        except np.linalg.LinAlgError:
            continue
        for d in range(n_basis):
            latent[lot_mask, d] = float(coeffs[d])
    return pd.DataFrame(
        {f"latent_state_{d + 1}": latent[:, d] for d in range(n_dims)},
        index=df_meta.index,
    )


# ============================================================
# Stage 1: PreMET inference (adapted from tabpfn_simple_probalistic.py)
# ============================================================

def run_stage1_premet(
    df: pd.DataFrame,
    dataset_name: str,
    *,
    output_dir: str,
    target_col: str,
    time_col: str,
    slot_col: str,
    lot_col: str,
    wafer_id_col: str,
    reference_slot_ids: list[int],
    train_ratio: float,
    val_ratio: float,
    model_path: str,
    n_estimators: int,
    softmax_temperature: float,
    average_before_softmax: bool,
    poly_features: int,
    subsample_samples: int,
    predict_batch_size: int,
    max_features: int,
    max_missing_ratio: float,
    min_variance: float,
    ci_quantile_lower: float,
    ci_quantile_upper: float,
    conf_width_thresholds: list[float],
    coverage_thresholds: list[float],
    temporal_lot_window_k: int = DEFAULT_TEMPORAL_LOT_WINDOW_K,
    residual_pca_components: int = DEFAULT_RESIDUAL_PCA_COMPONENTS,
    learn_lot_state: bool = DEFAULT_LEARN_LOT_STATE,
    lot_state_dims: int = DEFAULT_LOT_STATE_DIMS,
) -> Optional[dict]:
    """Run Stage-1 PreMET prediction.

    Returns a dict with keys:
      - premet_metrics: coverage/CI metrics on test non-ref wafers
      - y_test: actual premet values on full test set
      - y_pred: predicted premet on full test set
      - q_lower, q_upper: CI bounds on full test set
      - df_test: test rows of the original dataframe
      - val_end: index split position
      - test_is_ref: bool array for reference-slot rows in test
    """
    required = [target_col, slot_col, time_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ⚠️  skip {dataset_name}: missing columns {missing}")
        return None

    df = df.sort_values(time_col, ascending=True).reset_index(drop=True)

    if lot_col in df.columns:
        lot_ids = df[lot_col].astype(str)
    elif wafer_id_col in df.columns:
        lot_ids = df[wafer_id_col].astype(str).str[:-2]
        df[lot_col] = lot_ids
    else:
        print(f"  ⚠️  skip {dataset_name}: need '{lot_col}' or '{wafer_id_col}'")
        return None

    n_total = len(df)
    if n_total < 50:
        print(f"  ⚠️  skip {dataset_name}: too small n={n_total}")
        return None

    val_end = int(n_total * val_ratio)

    y = df[target_col].astype(float).to_numpy(dtype=np.float32)

    train_df = df.iloc[:val_end].copy()
    global_ref_model = fit_global_reference_model(
        train_df,
        target_col=target_col,
        slot_col=slot_col,
        lot_col=lot_col,
        reference_slot_ids=reference_slot_ids,
        n_residual_components=residual_pca_components,
    )

    X_raw = build_slot_ref_features(
        df, target_col=target_col, slot_col=slot_col,
        lot_col=lot_col, reference_slot_ids=reference_slot_ids,
    )
    X_raw = append_global_reference_features(
        df, X_raw, target_col=target_col, slot_col=slot_col,
        lot_col=lot_col, reference_slot_ids=reference_slot_ids,
        global_ref_model=global_ref_model,
    )
    if temporal_lot_window_k > 0:
        X_temporal = build_temporal_lot_features(
            df, time_col=time_col, lot_col=lot_col, target_col=target_col,
            slot_col=slot_col, reference_slot_ids=reference_slot_ids,
            window_k=temporal_lot_window_k,
        )
        X_raw = pd.concat([X_raw, X_temporal], axis=1)

    X_raw = _coerce_mixed_columns_for_tabpfn(X_raw)
    X_train_for_select = X_raw.iloc[:val_end]
    X_selected, selected_cols, fs_info = fast_select_features(
        X_train=X_train_for_select,
        y_train=y[:val_end],
        X_all=X_raw,
        max_features=max_features,
        max_missing_ratio=max_missing_ratio,
        min_variance=min_variance,
    )

    slots = df[slot_col].to_numpy()
    test_is_ref = np.isin(slots[val_end:], reference_slot_ids)
    if (~test_is_ref).sum() == 0:
        print(f"  ⚠️  skip {dataset_name}: no non-ref rows in test")
        return None

    print(
        f"  [Stage-1] features: {fs_info['raw_features']} → "
        f"miss={fs_info['after_missing_filter']} → "
        f"var={fs_info['after_variance_filter']} → "
        f"final={fs_info['after_score_filter']}"
    )

    t0 = time.time()
    model = create_model(
        model_path=model_path,
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        average_before_softmax=average_before_softmax,
        poly_features=poly_features,
        subsample_samples=subsample_samples,
    )
    model.fit(X_selected.iloc[:val_end], y[:val_end])
    t_fit = time.time() - t0

    if learn_lot_state and lot_state_dims > 0 and len(reference_slot_ids) > 0:
        df_meta_full = pd.DataFrame(
            {lot_col: lot_ids.to_numpy(), slot_col: slots}, index=df.index
        )
        latent_df = fit_lot_latent_states(
            model=model, X_selected=X_selected, y=y, df_meta=df_meta_full,
            lot_col=lot_col, slot_col=slot_col,
            reference_slot_ids=reference_slot_ids, n_dims=lot_state_dims,
        )
        X_selected = pd.concat([X_selected, latent_df.astype(np.float32)], axis=1)
        del model
        force_cleanup(light=True)
        model = create_model(
            model_path=model_path, n_estimators=n_estimators,
            softmax_temperature=softmax_temperature,
            average_before_softmax=average_before_softmax,
            poly_features=poly_features, subsample_samples=subsample_samples,
        )
        model.fit(X_selected.iloc[:val_end], y[:val_end])

    t1 = time.time()
    ci_quantiles = [ci_quantile_lower, ci_quantile_upper]
    y_pred_raw, quantile_arrays = predict_maybe_batched_with_quantiles(
        model, X_selected.iloc[val_end:], batch_size=predict_batch_size,
        ci_quantiles=ci_quantiles,
    )
    q_lower_raw, q_upper_raw = quantile_arrays[0], quantile_arrays[1]
    t_pred = time.time() - t1

    del model
    force_cleanup(light=True)

    meta_test = pd.DataFrame(
        {lot_col: lot_ids.iloc[val_end:].to_numpy(), slot_col: slots[val_end:]}
    )
    y_test = y[val_end:]
    y_pred = apply_residual_compensation(
        df_meta=meta_test, y_true=y_test, y_pred=y_pred_raw,
        lot_col=lot_col, slot_col=slot_col, reference_slot_ids=reference_slot_ids,
    )
    bias_shift = y_pred - y_pred_raw
    q_lower = q_lower_raw + bias_shift
    q_upper = q_upper_raw + bias_shift

    test_nonref = ~test_is_ref
    n_test_total = len(y_test)
    n_test_nonref = int(test_nonref.sum())

    premet_metrics = eval_metrics_prob(
        y_true=y_test[test_nonref],
        y_pred=y_pred[test_nonref],
        q_lower=q_lower[test_nonref],
        q_upper=q_upper[test_nonref],
        conf_width_thresholds=conf_width_thresholds,
        n_total_test=n_test_total,
    )
    coverage_metrics = eval_metrics_by_coverage(
        y_true=y_test[test_nonref],
        y_pred=y_pred[test_nonref],
        q_lower=q_lower[test_nonref],
        q_upper=q_upper[test_nonref],
        coverage_thresholds=coverage_thresholds,
        n_total_test=n_test_total,
    )
    premet_metrics.update(coverage_metrics)

    # Plot
    safe = dataset_name.replace("/", "_").replace(" ", "_").replace(".", "_")
    plot_path = os.path.join(output_dir, f"{safe}_premet_timeseries.png")
    ci_label = f"CI[{ci_quantile_lower:.0%},{ci_quantile_upper:.0%}]"
    plot_pred_true_timeseries(
        y_test=y_test, y_pred=y_pred, test_is_ref=test_is_ref,
        title=(
            f"{dataset_name} PreMET | Non-ref MAE={premet_metrics['mae']:.4f} "
            f"R²={premet_metrics['r2']:.4f} "
            f"Acc@0.5={premet_metrics['acc05']:.1f}% "
            f"CI-width={premet_metrics['ci_width_mean']:.3f} {ci_label}"
        ),
        out_path=plot_path, ylabel=target_col,
        q_lower=q_lower, q_upper=q_upper,
    )

    print(
        f"  [Stage-1] fit={t_fit:.1f}s pred={t_pred:.1f}s | "
        f"non-ref MAE={premet_metrics['mae']:.4f} R²={premet_metrics['r2']:.4f} "
        f"Acc@0.5={premet_metrics['acc05']:.1f}% Acc@1.0={premet_metrics['acc10']:.1f}%"
    )

    return {
        "premet_metrics": premet_metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "q_lower": q_lower,
        "q_upper": q_upper,
        "df_test": df.iloc[val_end:].reset_index(drop=True),
        "val_end": val_end,
        "test_is_ref": test_is_ref,
        "n_test_total": n_test_total,
        "n_test_nonref": n_test_nonref,
        "plot": plot_path,
        "time_sec": float(t_fit + t_pred),
        "n_features_raw": int(fs_info["raw_features"]),
        "n_features_used": int(len(selected_cols)),
    }


# ============================================================
# Stage 2: PostMET model training + inference
# ============================================================

def _safe_label_encode(le: LabelEncoder, values: np.ndarray) -> np.ndarray:
    """Transform values with LabelEncoder, mapping unseen labels to 0."""
    known = set(le.classes_)
    mapped = np.where(np.isin(values, list(known)), values, le.classes_[0])
    return le.transform(mapped).astype(np.float32)


def train_postmet_model(
    df_postmet: pd.DataFrame,
    *,
    pre_met_col: str,
    post_met_col: str,
    tool_col: str,
    loop_count_col: str,
    slot_col: str,
    model_path: str,
    n_estimators: int,
    softmax_temperature: float,
    average_before_softmax: bool,
    poly_features: int,
    subsample_samples: int,
) -> tuple[TabPFNRegressor, LabelEncoder, LabelEncoder, float]:
    """Fit a PostMET model on ALL rows of df_postmet.

    Returns (model, le_tool, le_slot, train_time_sec).
    """
    # Feature engineering (same as fit_loop_premet_postmet_tabpfn.py)
    df = df_postmet.copy()
    df["delta_MET"] = df[post_met_col] - df[pre_met_col]

    required_cols = [tool_col, slot_col, pre_met_col, loop_count_col, "delta_MET"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(
                f"PostMET training data missing required column: '{c}'. "
                f"Available: {list(df.columns)}"
            )

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    print(f"  [Stage-2] PostMET training rows after dropna: {len(df)}")

    le_tool = LabelEncoder()
    le_slot = LabelEncoder()
    tool_enc = le_tool.fit_transform(df[tool_col].astype(str)).astype(np.float32)
    slot_enc = le_slot.fit_transform(df[slot_col].astype(str)).astype(np.float32)

    X_train = pd.DataFrame({
        "tool_encoded": tool_enc,
        "slot_encoded": slot_enc,
        "pre_MET": df[pre_met_col].astype(np.float32).values,
        "loop_count": df[loop_count_col].astype(np.float32).values,
    })
    y_train = df["delta_MET"].astype(np.float32).values

    model = create_model(
        model_path=model_path,
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        average_before_softmax=average_before_softmax,
        poly_features=poly_features,
        subsample_samples=subsample_samples,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"  [Stage-2] PostMET model fitted on {len(df)} rows in {t_train:.1f}s")
    return model, le_tool, le_slot, t_train


def infer_postmet(
    postmet_model: TabPFNRegressor,
    le_tool: LabelEncoder,
    le_slot: LabelEncoder,
    df_test: pd.DataFrame,
    pre_met_values: np.ndarray,
    loop_count_values: np.ndarray,
    *,
    tool_col: str,
    slot_col: str,
) -> np.ndarray:
    """Run postmet inference given custom pre_MET and loop_count arrays.

    pre_met_values and loop_count_values must be aligned with df_test rows.
    Returns predicted post_MET = pre_met_values + delta_MET_pred.
    """
    pre_met_values = np.asarray(pre_met_values, dtype=np.float32)
    loop_count_values = np.asarray(loop_count_values, dtype=np.float32)

    tool_enc = _safe_label_encode(le_tool, df_test[tool_col].astype(str).values)
    slot_enc = _safe_label_encode(le_slot, df_test[slot_col].astype(str).values)

    X_infer = pd.DataFrame({
        "tool_encoded": tool_enc,
        "slot_encoded": slot_enc,
        "pre_MET": pre_met_values,
        "loop_count": loop_count_values,
    })

    delta_pred = postmet_model.predict(X_infer).astype(np.float32)
    post_met_pred = pre_met_values + delta_pred
    return post_met_pred


def run_stage2_postmet(
    stage1_result: dict,
    df_postmet_train: pd.DataFrame,
    dataset_name: str,
    *,
    output_dir: str,
    premet_col_in_test: str,
    postmet_pre_met_col: str,
    postmet_post_met_col: str,
    postmet_tool_col: str,
    postmet_loop_count_col: str,
    postmet_slot_col: str,
    model_path: str,
    n_estimators: int,
    softmax_temperature: float,
    average_before_softmax: bool,
    poly_features: int,
    subsample_samples: int,
) -> Optional[dict]:
    """Run Stage-2 PostMET.

    No actual post_MET ground truth is required in the PreMET test set.
    Two post_MET curves are predicted by the Stage-2 model and compared:

      • Scenario A (reference): pre_MET = Stage-1 GROUNDTRUTH PreMET,
        loop_count = ocd_to_loop(groundtruth PreMET)
      • Scenario B (deployed):  pre_MET = Stage-1 PREDICTED  PreMET,
        loop_count = ocd_to_loop(predicted  PreMET)

    Metrics measure the agreement of Scenario B (deployed) against
    Scenario A (groundtruth reference).
    """
    df_test = stage1_result["df_test"]
    y_pred_premet = stage1_result["y_pred"]   # Stage-1 predicted premet

    # Stage-2 only needs tool + slot from the test set; pre_MET and loop_count
    # are derived from Stage-1 groundtruth / predicted PreMET, so the actual
    # post_MET column is NOT required here.
    required_test_cols = [premet_col_in_test, postmet_tool_col, postmet_slot_col]
    missing = [c for c in required_test_cols if c not in df_test.columns]
    if missing:
        print(
            f"  ⚠️  [Stage-2] skip: premet test set missing columns {missing}. "
            f"Stage-2 requires the premet dataset to contain {required_test_cols}"
        )
        return None

    # Groundtruth PreMET (= Stage-1 target values) and Stage-1 predicted PreMET
    pre_met_actual = df_test[premet_col_in_test].astype(np.float32).values
    pre_met_predicted = y_pred_premet.astype(np.float32)

    valid_infer = np.isfinite(pre_met_actual) & np.isfinite(pre_met_predicted)
    if valid_infer.sum() < 10:
        print(f"  ⚠️  [Stage-2] skip: too few valid rows ({valid_infer.sum()}) in test set.")
        return None
    if valid_infer.sum() < len(valid_infer):
        print(
            f"  [Stage-2] Warning: {(~valid_infer).sum()} test rows have NaN in "
            "pre_MET inputs — excluding from Stage-2 inference."
        )

    df_test_valid = df_test[valid_infer].reset_index(drop=True)
    pre_met_actual = pre_met_actual[valid_infer]
    pre_met_predicted = pre_met_predicted[valid_infer]

    # Separately derive loop_count from groundtruth and predicted PreMET (OCD → loop)
    loop_count_actual = ocd_to_loop(pre_met_actual, out_of_range="clip").astype(int)
    loop_count_predicted = ocd_to_loop(pre_met_predicted, out_of_range="clip").astype(int)
    print(
        f"  [Stage-2] loop_count from GROUNDTRUTH PreMET: "
        f"{dict(pd.Series(loop_count_actual).value_counts().sort_index())}"
    )
    print(
        f"  [Stage-2] loop_count from PREDICTED  PreMET: "
        f"{dict(pd.Series(loop_count_predicted).value_counts().sort_index())}"
    )

    # Train postmet model on ALL postmet training data
    postmet_model, le_tool, le_slot, train_time = train_postmet_model(
        df_postmet_train,
        pre_met_col=postmet_pre_met_col,
        post_met_col=postmet_post_met_col,
        tool_col=postmet_tool_col,
        loop_count_col=postmet_loop_count_col,
        slot_col=postmet_slot_col,
        model_path=model_path,
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        average_before_softmax=average_before_softmax,
        poly_features=poly_features,
        subsample_samples=subsample_samples,
    )

    t0 = time.time()
    # Scenario A (reference): groundtruth PreMET + loop_count(groundtruth)
    y_post_from_gt = infer_postmet(
        postmet_model, le_tool, le_slot, df_test_valid,
        pre_met_values=pre_met_actual,
        loop_count_values=loop_count_actual,
        tool_col=postmet_tool_col,
        slot_col=postmet_slot_col,
    )
    # Scenario B (deployed): predicted PreMET + loop_count(predicted)
    y_post_from_pred = infer_postmet(
        postmet_model, le_tool, le_slot, df_test_valid,
        pre_met_values=pre_met_predicted,
        loop_count_values=loop_count_predicted,
        tool_col=postmet_tool_col,
        slot_col=postmet_slot_col,
    )
    t_infer = time.time() - t0

    del postmet_model
    force_cleanup(light=True)

    # Compare the two predicted post_MET curves. Scenario A (groundtruth-PreMET)
    # is the reference ("true"); Scenario B (predicted-PreMET) is the prediction.
    metrics = eval_metrics(y_post_from_gt, y_post_from_pred)
    loop_metrics = eval_metrics_by_loop_count(
        y_post_from_gt, y_post_from_pred, loop_count_actual
    )

    safe = dataset_name.replace("/", "_").replace(" ", "_").replace(".", "_")
    plot_ts_path = os.path.join(output_dir, f"{safe}_postmet_timeseries.png")
    plot_scatter_path = os.path.join(output_dir, f"{safe}_postmet_scatter.png")

    plot_postmet_timeseries(
        y_post_from_gt=y_post_from_gt,
        y_post_from_pred=y_post_from_pred,
        metrics=metrics,
        out_path=plot_ts_path,
        dataset_name=dataset_name,
    )
    plot_postmet_scatter(
        y_post_from_gt=y_post_from_gt,
        y_post_from_pred=y_post_from_pred,
        metrics=metrics,
        loop_counts=loop_count_actual,
        loop_metrics=loop_metrics,
        out_path=plot_scatter_path,
        dataset_name=dataset_name,
    )

    print(
        f"  [Stage-2] infer={t_infer:.1f}s | "
        f"PostMET agreement (predicted-PreMET vs groundtruth-PreMET): "
        f"MAE={metrics['mae']:.4f} R²={metrics['r2']:.4f} "
        f"Acc@0.5={metrics['acc05']:.1f}% Acc@1.0={metrics['acc10']:.1f}%"
    )

    return {
        "postmet_metrics": metrics,
        "loop_count_metrics": loop_metrics,
        "plot_timeseries": plot_ts_path,
        "plot_scatter": plot_scatter_path,
        "n_test": int(valid_infer.sum()),
        "train_time_sec": float(train_time),
        "infer_time_sec": float(t_infer),
    }

# ============================================================
# CLI helpers
# ============================================================

def _parse_reference_slot_ids(s: str) -> list[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Two-stage TabPFN: Stage-1 PreMET probabilistic prediction → "
            "Stage-2 PostMET prediction comparing actual vs predicted PreMET input."
        )
    )

    # Data paths
    p.add_argument(
        "--premet-data-path", type=str, default=DEFAULT_PREMET_DATA_PATH,
        help="Input file or folder (csv/parquet) for Stage-1 PreMET prediction.",
    )
    p.add_argument(
        "--postmet-data-path", type=str, default=DEFAULT_POSTMET_DATA_PATH,
        help=(
            "Single csv/parquet file with PostMET training data. "
            "ALL rows are used to fit the Stage-2 model. "
            "If empty, Stage-2 is skipped."
        ),
    )
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    # Stage-1 PreMET columns
    p.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL,
                   help="PreMET target column name (also used as pre_MET for Stage-2 Scenario A).")
    p.add_argument("--time-col", type=str, default=DEFAULT_TIME_COL)
    p.add_argument("--slot-col", type=str, default=DEFAULT_SLOT_COL)
    p.add_argument("--lot-col", type=str, default=DEFAULT_LOT_COL)
    p.add_argument("--wafer-id-col", type=str, default=DEFAULT_WAFER_ID_COL)
    p.add_argument(
        "--reference-slot-ids", type=str, default=DEFAULT_REFERENCE_SLOT_IDS,
        help='Comma-separated reference slot ids, e.g. "2,3,4,5,12,13,20,21,22,23".',
    )
    p.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    p.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)

    # Stage-2 PostMET columns
    p.add_argument(
        "--postmet-pre-met-col", type=str, default=DEFAULT_POSTMET_PRE_MET_COL,
        help=(
            "Pre-MET column name in both the PostMET training data and the "
            "PreMET test set. Used as the input feature 'pre_MET' for Stage-2."
        ),
    )
    p.add_argument(
        "--postmet-post-met-col", type=str, default=DEFAULT_POSTMET_POST_MET_COL,
        help="Post-MET column name (target) in the PostMET training data and PreMET test set.",
    )
    p.add_argument(
        "--postmet-tool-col", type=str, default=DEFAULT_POSTMET_TOOL_COL,
        help="Tool-name column in the PostMET training data and PreMET test set.",
    )
    p.add_argument(
        "--postmet-loop-count-col", type=str, default=DEFAULT_POSTMET_LOOP_COUNT_COL,
        help="Loop-count column in the PostMET training data and PreMET test set.",
    )
    p.add_argument(
        "--postmet-slot-col", type=str, default=DEFAULT_POSTMET_SLOT_COL,
        help="Slot-id column in the PostMET training data.",
    )

    # Model
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    p.add_argument("--softmax-temperature", type=float, default=DEFAULT_SOFTMAX_TEMPERATURE)
    p.add_argument(
        "--average-before-softmax", action="store_true",
        default=DEFAULT_AVERAGE_BEFORE_SOFTMAX,
    )
    p.add_argument("--poly-features", type=int, default=DEFAULT_POLY_FEATURES)
    p.add_argument("--subsample-samples", type=int, default=DEFAULT_SUBSAMPLE_SAMPLES)
    p.add_argument(
        "--predict-batch-size", type=int, default=DEFAULT_PREDICT_BATCH_SIZE,
        help="0 = predict all test rows at once.",
    )

    # Feature selection
    p.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    p.add_argument("--max-missing-ratio", type=float, default=DEFAULT_MAX_MISSING_RATIO)
    p.add_argument("--min-variance", type=float, default=DEFAULT_MIN_VARIANCE)

    # Advanced Stage-1 options
    p.add_argument("--temporal-lot-window-k", type=int, default=DEFAULT_TEMPORAL_LOT_WINDOW_K)
    p.add_argument("--residual-pca-components", type=int, default=DEFAULT_RESIDUAL_PCA_COMPONENTS)
    p.add_argument("--learn-lot-state", action="store_true", default=DEFAULT_LEARN_LOT_STATE)
    p.add_argument("--lot-state-dims", type=int, default=DEFAULT_LOT_STATE_DIMS)

    # CI / coverage
    p.add_argument("--ci-quantile-lower", type=float, default=DEFAULT_CI_QUANTILE_LOWER)
    p.add_argument("--ci-quantile-upper", type=float, default=DEFAULT_CI_QUANTILE_UPPER)
    p.add_argument(
        "--conf-width-thresholds", type=str, default=DEFAULT_CONF_WIDTH_THRESHOLDS,
        help='Comma-separated CI-width thresholds, e.g. "0.5,1.0,1.5".',
    )
    p.add_argument(
        "--coverage-thresholds", type=str, default=DEFAULT_COVERAGE_THRESHOLDS,
        help='Comma-separated coverage fractions (0–1), e.g. "0.10,0.20,0.30".',
    )

    return p


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    reference_slot_ids = _parse_reference_slot_ids(args.reference_slot_ids)
    conf_width_thresholds = _parse_float_list(args.conf_width_thresholds)
    coverage_thresholds = _parse_float_list(args.coverage_thresholds)

    if not (0.0 <= args.ci_quantile_lower <= 1.0):
        raise ValueError(f"--ci-quantile-lower must be in [0, 1], got {args.ci_quantile_lower}")
    if not (0.0 <= args.ci_quantile_upper <= 1.0):
        raise ValueError(f"--ci-quantile-upper must be in [0, 1], got {args.ci_quantile_upper}")
    if args.ci_quantile_lower >= args.ci_quantile_upper:
        raise ValueError(
            f"--ci-quantile-lower ({args.ci_quantile_lower}) must be < "
            f"--ci-quantile-upper ({args.ci_quantile_upper})"
        )

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load PostMET training data (if provided)
    df_postmet_train: Optional[pd.DataFrame] = None
    if args.postmet_data_path and os.path.exists(args.postmet_data_path):
        print(f"\nLoading PostMET training data: {args.postmet_data_path}")
        df_postmet_train = load_single_file(args.postmet_data_path)
        print(f"  PostMET training rows: {len(df_postmet_train)}")
    else:
        if args.postmet_data_path:
            print(
                f"  ⚠️  PostMET data path not found: '{args.postmet_data_path}'. "
                "Stage-2 will be skipped."
            )
        else:
            print("  Stage-2 disabled (no --postmet-data-path provided).")

    # Discover PreMET files
    files = discover_files(args.premet_data_path)
    print(
        f"\nFound {len(files)} PreMET file(s). OUTPUT_DIR={args.output_dir}\n"
        f"CI interval: [{args.ci_quantile_lower:.0%}, {args.ci_quantile_upper:.0%}] | "
        f"conf-width thresholds: {conf_width_thresholds} | "
        f"coverage thresholds: {coverage_thresholds}"
    )

    all_results = []
    t_all = time.time()

    for i, fp in enumerate(files):
        name = os.path.basename(fp)
        print(f"\n{'='*70}")
        print(f"[{i + 1}/{len(files)}] {name}")
        print(f"{'='*70}")

        df = load_single_file(fp)
        print(f"  shape={df.shape}")

        dataset_result: dict = {"dataset": name}

        try:
            # ─────────────────── Stage 1: PreMET ─────────────────────────────
            s1 = run_stage1_premet(
                df,
                dataset_name=name,
                output_dir=args.output_dir,
                target_col=args.target_col,
                time_col=args.time_col,
                slot_col=args.slot_col,
                lot_col=args.lot_col,
                wafer_id_col=args.wafer_id_col,
                reference_slot_ids=reference_slot_ids,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                model_path=args.model_path,
                n_estimators=args.n_estimators,
                softmax_temperature=args.softmax_temperature,
                average_before_softmax=args.average_before_softmax,
                poly_features=args.poly_features,
                subsample_samples=args.subsample_samples,
                predict_batch_size=args.predict_batch_size,
                max_features=args.max_features,
                max_missing_ratio=args.max_missing_ratio,
                min_variance=args.min_variance,
                ci_quantile_lower=args.ci_quantile_lower,
                ci_quantile_upper=args.ci_quantile_upper,
                conf_width_thresholds=conf_width_thresholds,
                coverage_thresholds=coverage_thresholds,
                temporal_lot_window_k=args.temporal_lot_window_k,
                residual_pca_components=args.residual_pca_components,
                learn_lot_state=args.learn_lot_state,
                lot_state_dims=args.lot_state_dims,
            )

            if s1 is None:
                continue

            dataset_result["stage1"] = {
                "n_rows": int(len(df)),
                "n_test": s1["n_test_total"],
                "n_test_nonref": s1["n_test_nonref"],
                "n_features_raw": s1["n_features_raw"],
                "n_features_used": s1["n_features_used"],
                "time_sec": s1["time_sec"],
                "premet_metrics": s1["premet_metrics"],
                "plot": s1["plot"],
            }

            # Print Stage-1 detail
            m = s1["premet_metrics"]
            print(
                f"\n  ── Stage-1 PreMET Results ──\n"
                f"  Non-ref coverage: {s1['n_test_nonref']}/{s1['n_test_total']} "
                f"({s1['n_test_nonref']/s1['n_test_total']*100:.1f}%)\n"
                f"  MAE={m['mae']:.4f}  R²={m['r2']:.4f}  "
                f"Acc@0.5={m['acc05']:.1f}%  Acc@1.0={m['acc10']:.1f}%"
            )
            print(f"  CI-width thresholds:")
            for thr in conf_width_thresholds:
                key = f"ci_thr{thr:.1f}"
                cov_tot = m.get(f"{key}_coverage_pct_of_total", float("nan"))
                thr_mae = m.get(f"{key}_mae", float("nan"))
                thr_r2 = m.get(f"{key}_r2", float("nan"))
                thr_acc05 = m.get(f"{key}_acc05", float("nan"))
                print(
                    f"    ≤{thr:.1f}: coverage={cov_tot:.1f}% of total  "
                    f"MAE={thr_mae:.4f}  R²={thr_r2:.4f}  Acc@0.5={thr_acc05:.1f}%"
                )
            print(f"  Coverage-based selection:")
            for cov_pct in coverage_thresholds:
                key = f"cov_{cov_pct*100:.0f}pct"
                if key in m:
                    ci = m[key]
                    print(
                        f"    Top {cov_pct*100:.0f}% non-ref: n={ci['n_samples']} "
                        f"({ci['coverage_pct_of_total']:.1f}% of total)  "
                        f"MAE={ci['mae']:.4f}  R²={ci['r2']:.4f}  "
                        f"Acc@0.5={ci['acc05']:.1f}%  "
                        f"CI-width={ci['ci_width_mean']:.3f}"
                    )

            # ─────────────────── Stage 2: PostMET ────────────────────────────
            if df_postmet_train is not None:
                s2 = run_stage2_postmet(
                    stage1_result=s1,
                    df_postmet_train=df_postmet_train,
                    dataset_name=name,
                    output_dir=args.output_dir,
                    premet_col_in_test=args.target_col,
                    postmet_pre_met_col=args.postmet_pre_met_col,
                    postmet_post_met_col=args.postmet_post_met_col,
                    postmet_tool_col=args.postmet_tool_col,
                    postmet_loop_count_col=args.postmet_loop_count_col,
                    postmet_slot_col=args.postmet_slot_col,
                    model_path=args.model_path,
                    n_estimators=args.n_estimators,
                    softmax_temperature=args.softmax_temperature,
                    average_before_softmax=args.average_before_softmax,
                    poly_features=args.poly_features,
                    subsample_samples=args.subsample_samples,
                )

                if s2 is not None:
                    dataset_result["stage2"] = s2

                    m2 = s2["postmet_metrics"]
                    print(
                        f"\n  ── Stage-2 PostMET Results ──\n"
                        f"  n_test = {s2['n_test']}\n"
                        f"  PostMET agreement (predicted-PreMET vs groundtruth-PreMET):\n"
                        f"    MAE={m2['mae']:.4f}  R²={m2['r2']:.4f}  "
                        f"Acc@0.5={m2['acc05']:.1f}%  Acc@1.0={m2['acc10']:.1f}%"
                    )
                    if s2["loop_count_metrics"]:
                        print("  Loop-count metrics (grouped by groundtruth loop):")
                        for lk, lv in sorted(
                            s2["loop_count_metrics"].items(),
                            key=lambda kv: int(kv[0].split("_")[1]),
                        ):
                            print(
                                f"    {lk}: n={lv['n_samples']}  "
                                f"MAE={lv['mae']:.4f}  R²={lv['r2']:.4f}  "
                                f"Acc@0.5={lv['acc05']:.1f}%"
                            )

            all_results.append(dataset_result)

            # Save per-dataset JSON
            safe = name.replace("/", "_").replace(" ", "_").replace(".", "_")
            json_path = os.path.join(args.output_dir, f"{safe}_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(dataset_result, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n  JSON saved: {json_path}")

        except Exception as e:
            print(f"  ❌ failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del df
            force_cleanup(light=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_time = time.time() - t_all
    print(f"\n{'='*70}")
    print(f"Done. processed={len(all_results)}/{len(files)}  total_time={total_time:.1f}s")
    print(f"{'='*70}\n")

    s1_results = [r for r in all_results if "stage1" in r]
    if s1_results:
        avg_mae = float(np.mean([r["stage1"]["premet_metrics"]["mae"] for r in s1_results]))
        avg_r2 = float(np.mean([r["stage1"]["premet_metrics"]["r2"] for r in s1_results]))
        avg_acc05 = float(np.mean([r["stage1"]["premet_metrics"]["acc05"] for r in s1_results]))
        avg_acc10 = float(np.mean([r["stage1"]["premet_metrics"]["acc10"] for r in s1_results]))
        print(
            f"[Stage-1 PreMET Summary]\n"
            f"  AVG MAE={avg_mae:.4f}  AVG R²={avg_r2:.4f}  "
            f"AVG Acc@0.5={avg_acc05:.1f}%  AVG Acc@1.0={avg_acc10:.1f}%\n"
        )

    s2_results = [r for r in all_results if "stage2" in r]
    if s2_results:
        avg_mae = float(np.mean([r["stage2"]["postmet_metrics"]["mae"] for r in s2_results]))
        avg_r2 = float(np.mean([r["stage2"]["postmet_metrics"]["r2"] for r in s2_results]))
        avg_acc05 = float(np.mean([r["stage2"]["postmet_metrics"]["acc05"] for r in s2_results]))
        avg_acc10 = float(np.mean([r["stage2"]["postmet_metrics"]["acc10"] for r in s2_results]))
        print(
            f"[Stage-2 PostMET Summary]\n"
            f"  PostMET agreement (predicted-PreMET vs groundtruth-PreMET):\n"
            f"  AVG MAE={avg_mae:.4f}  AVG R²={avg_r2:.4f}  "
            f"AVG Acc@0.5={avg_acc05:.1f}%  AVG Acc@1.0={avg_acc10:.1f}%\n"
        )


if __name__ == "__main__":
    main()
    print("Process finished, exiting now...", flush=True)
    # Use os._exit to bypass Python atexit handlers, which prevents occasional
    # CUDA / torch teardown hangs when running on GPU (matches source scripts).
    os._exit(0)
