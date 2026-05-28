import os
os.environ["TABPFN_NO_TELEMETRY"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["DISABLE_POSTHOG"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["SEGMENT_WRITE_KEY"] = ""
os.environ["ANALYTICS_DISABLED"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

import glob
import time
import gc
import warnings
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import f_regression

from tabpfn import TabPFNRegressor
# import posthog
# posthog.disabled = True

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Degrees of freedom")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")


# ============================================================
# Defaults
# ============================================================

DEFAULT_DATA_PATH = "/ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB01_CHA1_1011_1229.parquet"
DEFAULT_OUTPUT_DIR = "./results/EPLBAB01_CHA1_1101_1120_simple_fast"

DEFAULT_TARGET_COL = "met"
DEFAULT_TIME_COL = "start_time"
DEFAULT_SLOT_COL = "slot_id"
DEFAULT_LOT_COL = "lot_id"
DEFAULT_WAFER_ID_COL = "wafer_id"

DEFAULT_REFERENCE_SLOT_IDS = "2,3,4,5,12,13,20,21,22,23"

DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.8

DEFAULT_MODEL_PATH = "/ossfs/workspace/xrfm/TabPFN-main/models/tabpfn-v2.5-regressor-v2.5_default.ckpt"

# ===== 最大提速导向默认值 =====
DEFAULT_N_ESTIMATORS = 4
DEFAULT_SOFTMAX_TEMPERATURE = 0.9
DEFAULT_AVERAGE_BEFORE_SOFTMAX = True

DEFAULT_POLY_FEATURES = 1
DEFAULT_SUBSAMPLE_SAMPLES = 2048
DEFAULT_PREDICT_BATCH_SIZE = 0   # 0 = whole test set once

# 特征筛选
DEFAULT_MAX_FEATURES = 120
DEFAULT_MAX_MISSING_RATIO = 0.60
DEFAULT_MIN_VARIANCE = 1e-10

# Learnable prior-state features
DEFAULT_TEMPORAL_LOT_WINDOW_K = 5      # Approach 1: temporal drift window size
DEFAULT_RESIDUAL_PCA_COMPONENTS = 2    # Approach 3: residual PCA component count
DEFAULT_LEARN_LOT_STATE = False        # Approach 2: enable per-lot latent state vectors
DEFAULT_LOT_STATE_DIMS = 2             # Approach 2: latent state dimensionality K

# 概率区间 / 置信度阈值
DEFAULT_CI_QUANTILE_LOWER = 0.1   # 80% 预测区间下界分位数
DEFAULT_CI_QUANTILE_UPPER = 0.9   # 80% 预测区间上界分位数
DEFAULT_CONF_WIDTH_THRESHOLDS = "1.0,2.0,3.0"  # CI 宽度阈值列表 (MET 原始单位)


# ============================================================
# IO
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
# Metrics / Plot
# ============================================================

def acc_within(y_true: np.ndarray, y_pred: np.ndarray, thr: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= thr) * 100.0)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
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
) -> dict:
    """Extended metrics including prediction-interval statistics and threshold-based accuracy.

    For each threshold in *conf_width_thresholds*, a "high-confidence" subset is defined as
    samples whose CI width (q_upper - q_lower) is at most the threshold.  Standard regression
    metrics are then computed on that subset, together with the coverage fraction.
    """
    base = eval_metrics(y_true, y_pred)

    ci_width = q_upper - q_lower
    empirical_coverage = float(np.mean((y_true >= q_lower) & (y_true <= q_upper)) * 100.0)

    base.update({
        "ci_width_mean": float(np.mean(ci_width)),
        "ci_width_median": float(np.median(ci_width)),
        "ci_empirical_coverage_pct": empirical_coverage,
    })

    for thr in conf_width_thresholds:
        high_conf = ci_width <= thr
        coverage_pct = float(np.mean(high_conf) * 100.0)
        key_prefix = f"ci_thr{thr:.1f}"
        base[f"{key_prefix}_coverage_pct"] = coverage_pct
        if high_conf.sum() > 0:
            base[f"{key_prefix}_mae"] = float(mean_absolute_error(y_true[high_conf], y_pred[high_conf]))
            base[f"{key_prefix}_r2"] = float(r2_score(y_true[high_conf], y_pred[high_conf]))
            base[f"{key_prefix}_acc05"] = float(acc_within(y_true[high_conf], y_pred[high_conf], 0.5))
            base[f"{key_prefix}_acc10"] = float(acc_within(y_true[high_conf], y_pred[high_conf], 1.0))
        else:
            base[f"{key_prefix}_mae"] = float("nan")
            base[f"{key_prefix}_r2"] = float("nan")
            base[f"{key_prefix}_acc05"] = float("nan")
            base[f"{key_prefix}_acc10"] = float("nan")

    return base


def plot_pred_true_timeseries(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_is_ref: np.ndarray,
    title: str,
    out_path: str,
    ylabel: str,
    q_lower: np.ndarray | None = None,
    q_upper: np.ndarray | None = None,
) -> None:
    x = np.arange(len(y_test))
    is_nonref = ~test_is_ref

    plt.figure(figsize=(18, 6))

    plt.fill_between(
        x,
        y_test - 0.5,
        y_test + 0.5,
        alpha=0.10,
        color="green",
        label="±0.5 band",
    )

    # Prediction interval (CI) band for non-ref samples
    if q_lower is not None and q_upper is not None:
        plt.fill_between(
            x[is_nonref],
            q_lower[is_nonref],
            q_upper[is_nonref],
            alpha=0.18,
            color="steelblue",
            label="pred CI (non-ref)",
        )

    plt.plot(x, y_test, color="black", alpha=0.35, linewidth=1.0, label="true (all)")
    plt.scatter(x[is_nonref], y_test[is_nonref], s=8, color="black", alpha=0.6, label="true (non-ref)")
    plt.scatter(x[test_is_ref], y_test[test_is_ref], s=8, color="gray", alpha=0.4, label="true (ref)")

    plt.plot(x, y_pred, color="steelblue", alpha=0.55, linewidth=1.2, label="pred (comp)")
    plt.scatter(x[is_nonref], y_pred[is_nonref], s=8, color="steelblue", alpha=0.6, label="pred (comp, non-ref)")
    plt.scatter(x[test_is_ref], y_pred[test_is_ref], s=8, color="salmon", alpha=0.4, label="pred (comp, ref)")

    plt.title(title)
    plt.xlabel("test sample index (time order)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


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
        lot_nonref_mask = lot_mask & (~is_ref)

        if lot_ref_mask.sum() == 0:
            continue

        bias = np.nanmean(y_true[lot_ref_mask] - y_pred[lot_ref_mask])
        if np.isnan(bias):
            continue

        compensated[lot_nonref_mask] += bias

    return compensated


# ============================================================
# Slot / reference-MET feature engineering
# ============================================================

def build_slot_ref_features(
    df: pd.DataFrame,
    *,
    target_col: str,
    slot_col: str,
    lot_col: str,
    reference_slot_ids: list[int],
) -> pd.DataFrame:
    """Build a feature matrix using only slot-position and reference-wafer MET data.

    No FDC sensor columns are used.  For each wafer row the following groups are
    constructed:

    1. **Slot-position features** – normalized position, centre distance, polynomial
       trend terms, and a binary flag for reference slots.
    2. **Lot-level reference-MET aggregates** – mean, std, median, min, max, range
       and count of reference-wafer METs within the same lot.  Reference wafers use
       leave-one-out aggregation to avoid target leakage.
    3. **Per-reference-slot MET values** (wide format) – ``ref_slot_{i}_met`` for
       every slot id in *reference_slot_ids*, taken from the same lot.  A reference
       wafer's own slot column is set to NaN (leave-one-out).
    4. **Deviation features** – each wide-format value minus the lot mean.
    5. **Interpolated reference MET** – piecewise-linear interpolation of reference
       METs at the current slot position.
    """
    slots = df[slot_col].to_numpy(dtype=np.int32)
    lots = df[lot_col].to_numpy()
    mets = df[target_col].to_numpy(dtype=np.float32)
    n_rows = len(df)

    ref_set = set(reference_slot_ids)
    is_ref = np.isin(slots, reference_slot_ids)

    # ── 1. Slot-position features ─────────────────────────────────────────────
    slot_vals = df[slot_col].values
    slot_min = float(slot_vals.min())
    slot_max = float(slot_vals.max())
    slot_range = max(slot_max - slot_min, 1.0)

    slot_norm = (slots - slot_min) / slot_range        # [0, 1]
    slot_center_dist = np.abs(slot_norm - 0.5)         # [0, 0.5]

    feat: dict[str, np.ndarray] = {
        "slot_id": slots.astype(np.float32),
        "slot_norm": slot_norm.astype(np.float32),
        "slot_center_dist": slot_center_dist.astype(np.float32),
        "slot_trend_sq": (slot_norm ** 2).astype(np.float32),
        "slot_trend_cubic": (slot_norm ** 3).astype(np.float32),
        "is_ref_slot": is_ref.astype(np.float32),
    }

    # Nearest reference slot distance
    ref_ids_arr = np.array(sorted(ref_set), dtype=float)
    if len(ref_ids_arr) > 0:
        nearest_ref_dist = np.min(
            np.abs(slots[:, None].astype(float) - ref_ids_arr[None, :]), axis=1
        ).astype(np.float32)
    else:
        nearest_ref_dist = np.zeros(n_rows, dtype=np.float32)
    feat["nearest_ref_dist"] = nearest_ref_dist

    # ── 2. Lot-level reference-MET aggregates (leave-one-out for ref wafers) ──
    lot_ref_met_mean = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_std = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_median = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_min = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_max = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_range = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_count = np.zeros(n_rows, dtype=np.float32)

    # ── 3. Wide-format per-reference-slot MET values ──────────────────────────
    ref_slot_mets: dict[int, np.ndarray] = {
        sid: np.full(n_rows, np.nan, dtype=np.float32) for sid in reference_slot_ids
    }

    # ── 5. Interpolated reference MET at current slot ─────────────────────────
    ref_met_interp = np.full(n_rows, np.nan, dtype=np.float32)

    for lot in np.unique(lots):
        lot_mask = lots == lot
        lot_ref_mask = lot_mask & is_ref
        n_ref = int(lot_ref_mask.sum())
        if n_ref == 0:
            continue

        lot_ref_slots = slots[lot_ref_mask]
        lot_ref_mets = mets[lot_ref_mask]

        # slot -> MET dictionary for this lot's reference wafers
        slot_met_dict: dict[int, float] = {}
        for s, m in zip(lot_ref_slots.tolist(), lot_ref_mets.tolist()):
            slot_met_dict[s] = m

        # Fill wide-format per-slot MET
        for sid in reference_slot_ids:
            if sid in slot_met_dict:
                ref_slot_mets[sid][lot_mask] = slot_met_dict[sid]

        # Precompute full-lot reference stats
        total_sum = float(np.nansum(lot_ref_mets))
        total_sum2 = float(np.nansum(lot_ref_mets ** 2))

        # Sorted reference slots/METs for interpolation
        sort_order = np.argsort(lot_ref_slots)
        sorted_ref_slots = lot_ref_slots[sort_order].astype(float)
        sorted_ref_mets = lot_ref_mets[sort_order]

        # Non-reference rows: vectorised fill
        nonref_indices = np.where(lot_mask & ~is_ref)[0]
        if len(nonref_indices) > 0:
            mean_val = total_sum / n_ref
            var_val = max(total_sum2 / n_ref - mean_val ** 2, 0.0)
            lot_ref_met_mean[nonref_indices] = mean_val
            lot_ref_met_std[nonref_indices] = float(np.sqrt(var_val))
            lot_ref_met_median[nonref_indices] = float(np.nanmedian(lot_ref_mets))
            lot_ref_met_min[nonref_indices] = float(np.nanmin(lot_ref_mets))
            lot_ref_met_max[nonref_indices] = float(np.nanmax(lot_ref_mets))
            lot_ref_met_range[nonref_indices] = float(np.nanmax(lot_ref_mets) - np.nanmin(lot_ref_mets))
            lot_ref_met_count[nonref_indices] = float(n_ref)
            for idx in nonref_indices:
                ref_met_interp[idx] = float(
                    np.interp(slots[idx], sorted_ref_slots, sorted_ref_mets)
                )

        # Reference rows: leave-one-out fill
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
            # Interpolation using LOO reference slots/METs
            loo_items = [(s, slot_met_dict[s]) for s in slot_met_dict if s != curr_slot]
            loo_slots = np.array([s for s, _ in loo_items], dtype=float)
            loo_met_vals = np.array([m for _, m in loo_items], dtype=np.float32)
            if len(loo_slots) >= 1:
                order = np.argsort(loo_slots)
                ref_met_interp[idx] = float(
                    np.interp(float(curr_slot), loo_slots[order], loo_met_vals[order])
                )
            # Wide-format: own slot = NaN (leave-one-out)
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

    # Wide-format and deviation columns
    for sid in reference_slot_ids:
        feat[f"ref_slot_{sid}_met"] = ref_slot_mets[sid]
        feat[f"ref_slot_{sid}_met_dev"] = ref_slot_mets[sid] - lot_ref_met_mean

    result = pd.DataFrame(feat, index=df.index)

    # Fill remaining NaNs with column median (robustness for TabPFN)
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
    """Build cross-lot temporal drift features for each wafer row.

    For each lot, uses the reference-MET statistics of the ``window_k`` most
    recent **preceding** lots (strictly earlier by earliest ``time_col``) as
    proxy features for the unknown prior state of the wafer before this step.

    No data leakage: only lots whose earliest timestamp is strictly before the
    current lot's earliest timestamp are included.

    Returns a DataFrame aligned to ``df`` with columns:
        - ``prev_k_lots_ref_mean``   : mean of reference METs across the window
        - ``prev_k_lots_ref_std``    : std of reference METs across the window
        - ``prev_k_lots_ref_min``    : min lot-mean reference MET in the window
        - ``prev_k_lots_ref_max``    : max lot-mean reference MET in the window
        - ``prev_k_lots_ref_trend``  : linear slope of lot-mean METs over the window
        - ``prev_lot_ref_mean``      : reference MET mean of the immediately preceding lot
        - ``lot_rank_in_window``     : 0-based temporal rank of this lot
        - ``lot_time_gap_hours``     : hours between this lot's first sample and the
                                       previous lot's first sample
    """
    lots = df[lot_col].to_numpy()
    mets = df[target_col].to_numpy(dtype=np.float32)
    slots = df[slot_col].to_numpy()
    times = df[time_col].to_numpy()
    n_rows = len(df)
    is_ref = np.isin(slots, reference_slot_ids)

    # Compute per-lot: earliest timestamp and mean reference MET
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

    # Sort lots by earliest time to establish temporal order
    try:
        lots_sorted = sorted(unique_lots, key=lambda lo: lot_earliest_time[lo])
    except Exception:
        lots_sorted = list(unique_lots)

    lot_rank = {lot: i for i, lot in enumerate(lots_sorted)}

    # Output arrays
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

        # Preceding lots within the window (strictly before current lot)
        preceding = lots_sorted[max(0, rank - window_k): rank]
        if not preceding:
            continue

        # Reference MET means of preceding lots (only where available)
        prev_means = [
            lot_ref_mean[pl] for pl in preceding if not np.isnan(lot_ref_mean.get(pl, np.nan))
        ]
        if not prev_means:
            continue

        prev_arr = np.array(prev_means, dtype=np.float32)
        prev_k_ref_mean[lot_mask] = float(np.nanmean(prev_arr))
        prev_k_ref_std[lot_mask] = float(np.nanstd(prev_arr)) if len(prev_arr) > 1 else 0.0
        prev_k_ref_min[lot_mask] = float(np.nanmin(prev_arr))
        prev_k_ref_max[lot_mask] = float(np.nanmax(prev_arr))

        # Most recent preceding lot's reference mean
        prev_lot_ref_mean_arr[lot_mask] = float(lot_ref_mean.get(preceding[-1], np.nan))

        # Linear trend of lot-mean METs across the window
        if len(prev_arr) >= 2:
            x_t = np.arange(len(prev_arr), dtype=np.float32)
            x_c = x_t - x_t.mean()
            denom = float(np.dot(x_c, x_c))
            if denom > 0.0:
                slope = float(np.dot(x_c, prev_arr - float(prev_arr.mean())) / denom)
            else:
                slope = 0.0
            prev_k_ref_trend[lot_mask] = slope
        else:
            prev_k_ref_trend[lot_mask] = 0.0

        # Time gap between this lot and the immediately preceding lot (in hours)
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

    # Fill remaining NaNs with column median
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
) -> dict[object, np.ndarray]:
    slots = df[slot_col].to_numpy()
    lots = df[lot_col].to_numpy()
    mets = df[target_col].to_numpy(dtype=np.float32)

    is_ref = np.isin(slots, reference_slot_ids)
    ref_ids = list(reference_slot_ids)
    lot_profiles: dict[object, np.ndarray] = {}

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
) -> dict[str, np.ndarray]:
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
    if lot_profiles:
        mat = np.vstack(list(lot_profiles.values())).astype(np.float32)
    else:
        mat = np.empty((0, n_ref), dtype=np.float32)

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

            # Approach 3 (residual PCA): decompose variation unexplained by primary components
            n_resid_comp = min(int(n_residual_components), centered.shape[0], centered.shape[1])
            if n_resid_comp <= 0 or components.shape[0] == 0:
                residual_components = np.empty((0, n_ref), dtype=np.float32)
            else:
                try:
                    proj = centered @ components.T          # (n_lots, n_comp)
                    reconstruction = proj @ components      # (n_lots, n_ref)
                    resid_mat = centered - reconstruction   # unexplained variation
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
    global_ref_model: dict[str, np.ndarray] | None,
) -> pd.DataFrame:
    if global_ref_model is None or len(reference_slot_ids) == 0:
        return X_base

    template_profile = np.asarray(global_ref_model.get("template_profile", np.array([])), dtype=np.float32)
    slot_fill_values = np.asarray(global_ref_model.get("slot_fill_values", np.array([])), dtype=np.float32)
    components = np.asarray(global_ref_model.get("components", np.empty((0, 0))), dtype=np.float32)
    residual_components = np.asarray(
        global_ref_model.get("residual_components", np.empty((0, 0))), dtype=np.float32
    )
    model_ref_ids = np.asarray(
        global_ref_model.get("reference_slot_ids", np.asarray(reference_slot_ids, dtype=np.int32)),
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
    interp_template_profile = template_profile
    if len(interp_ref_ids) >= 2:
        sort_idx = np.argsort(interp_ref_ids)
        interp_ref_ids = interp_ref_ids[sort_idx]
        interp_template_profile = interp_template_profile[sort_idx]

    # Slot normalisation for cross-product features
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

        raw_profile = lot_profiles.get(lot, np.full(len(reference_slot_ids), np.nan, dtype=np.float32))
        filled_profile = np.where(np.isnan(raw_profile), slot_fill_values, raw_profile).astype(np.float32)
        centered_profile = filled_profile - template_profile

        profile_norm = float(np.linalg.norm(filled_profile))
        profile_centered = filled_profile - float(np.mean(filled_profile))
        profile_centered_norm = float(np.linalg.norm(profile_centered))

        cos_val = float(np.dot(filled_profile, template_profile) / (profile_norm * template_norm + eps))
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

        # Approach 3 (residual PCA): project onto residual PCA components
        if n_resid_comp > 0 and residual_components.shape[1] == len(reference_slot_ids):
            # Remove primary-component reconstruction before projecting
            if n_comp > 0:
                lot_resid_vec = centered_profile - (centered_profile @ components.T) @ components
            else:
                lot_resid_vec = centered_profile
            lot_resid_scores = lot_resid_vec @ residual_components.T
            resid_pc_scores[lot_idx, :] = lot_resid_scores[None, :]

        if len(interp_ref_ids) >= 2:
            global_ref_template_interp[lot_idx] = np.interp(slots[lot_idx], interp_ref_ids, interp_template_profile)
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
        X["global_ref_interp_resid"] = X["ref_met_interp"].to_numpy(dtype=np.float32) - global_ref_template_interp

    for j in range(n_comp):
        X[f"global_ref_pc{j + 1}"] = pc_scores[:, j]

    # Approach 3 (residual PCA): residual PCA scores and slot-position cross-products
    for j in range(n_resid_comp):
        col_name = f"global_ref_resid_pc{j + 1}"
        X[col_name] = resid_pc_scores[:, j]
        # Cross-product with normalised slot position captures position-dependent unexplained variation
        X[f"{col_name}_x_slot"] = resid_pc_scores[:, j] * slot_norm_all

    return X


def _coerce_mixed_columns_for_tabpfn(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # object列尽量转category，数值列转float32
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

    # 1) 缺失率过滤
    miss_ratio = X_train.isna().mean()
    keep_cols = miss_ratio[miss_ratio <= max_missing_ratio].index.tolist()
    if not keep_cols:
        keep_cols = cols
    X_train_1 = X_train[keep_cols]
    info["after_missing_filter"] = int(len(keep_cols))

    # 2) 只对数值列做打分筛选；非数值列暂时丢弃（最大提速优先）
    num_cols = X_train_1.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        # 如果一个数值列都没有，就退化成前max_features列
        selected = keep_cols[:max_features]
        info["after_variance_filter"] = int(len(selected))
        info["after_score_filter"] = int(len(selected))
        return X_all[selected], selected, info

    X_num = X_train_1[num_cols]

    # 3) 低方差过滤
    variances = X_num.var(axis=0, skipna=True)
    keep_num_cols = variances[variances > min_variance].index.tolist()
    if not keep_num_cols:
        keep_num_cols = num_cols
    X_num = X_num[keep_num_cols]
    info["after_variance_filter"] = int(len(keep_num_cols))

    # 4) 单变量打分，选 top-K
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
# TabPFN
# ============================================================

def create_model(
    model_path: str,
    n_estimators: int,
    softmax_temperature: float,
    average_before_softmax: bool,
    poly_features: int,
    subsample_samples: int,
) -> TabPFNRegressor:
    poly_features = max(1, int(poly_features))
    subsample_samples = max(256, int(subsample_samples))

    return TabPFNRegressor(
        model_path=model_path,
        device="cuda",
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        average_before_softmax=average_before_softmax,
        memory_saving_mode=True,
        ignore_pretraining_limits=True,
        inference_config={
            "SUBSAMPLE_SAMPLES": subsample_samples,
            "POLYNOMIAL_FEATURES": poly_features,
        },
    )


def predict_maybe_batched(model: TabPFNRegressor, X: pd.DataFrame, batch_size: int) -> np.ndarray:
    if batch_size is None or batch_size <= 0 or len(X) <= batch_size:
        return model.predict(X)

    preds = []
    for i in range(0, len(X), batch_size):
        preds.append(model.predict(X.iloc[i:i + batch_size]))
    return np.concatenate(preds)


def predict_maybe_batched_with_quantiles(
    model: TabPFNRegressor,
    X: pd.DataFrame,
    batch_size: int,
    ci_quantiles: list[float],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (mean_predictions, [q_lower_array, q_upper_array, ...]) using TabPFN quantile output."""
    def _predict_batch(batch: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
        result = model.predict(batch, output_type="main", quantiles=ci_quantiles)
        return result["mean"], result["quantiles"]

    if batch_size is None or batch_size <= 0 or len(X) <= batch_size:
        return _predict_batch(X)

    means: list[np.ndarray] = []
    quantile_parts: list[list[np.ndarray]] = [[] for _ in ci_quantiles]
    for i in range(0, len(X), batch_size):
        batch_mean, batch_quantiles = _predict_batch(X.iloc[i:i + batch_size])
        means.append(batch_mean)
        for j, q_arr in enumerate(batch_quantiles):
            quantile_parts[j].append(q_arr)

    return np.concatenate(means), [np.concatenate(parts) for parts in quantile_parts]


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
    """Fit per-lot latent state vectors from initial-model residuals on reference wafers.

    Uses the already-fit ``model`` to compute predictions for **all** rows, then
    for each lot performs a closed-form linear least-squares fit on the
    reference-wafer residuals (``y_ref - pred_ref``) to extract a K-dimensional
    latent state vector.  The K coefficients are broadcast to every wafer row in
    that lot as new features ``latent_state_1``…``latent_state_{n_dims}``.

    The basis functions are ``[1, slot_norm, slot_norm², …]`` up to ``n_dims``
    terms, giving:
        - ``latent_state_1`` – systematic bias (intercept of the residual model)
        - ``latent_state_2`` – slot-position trend of residuals (slope)
        - higher dims – higher-order polynomial coefficients

    For lots that have no reference wafers the latent states are set to 0.

    Returns a :class:`pandas.DataFrame` aligned to ``df_meta`` containing the
    ``n_dims`` latent state columns.
    """
    n_dims = max(1, int(n_dims))
    lots = df_meta[lot_col].to_numpy()
    slots = df_meta[slot_col].to_numpy(dtype=np.float32)
    is_ref = np.isin(slots, list(reference_slot_ids))
    n_rows = len(df_meta)

    # Predict all rows with the initial model (one pass)
    preds = model.predict(X_selected)

    # Normalise slot positions across the full dataset for consistent basis
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

        # Build basis matrix [1, s, s², …] for the reference rows
        n_basis = min(n_dims, n_ref_in_lot)
        phi_ref = np.column_stack([slot_norm_ref ** d for d in range(n_basis)])  # (n_ref, n_basis)

        try:
            coeffs, _, _, _ = np.linalg.lstsq(phi_ref, resid, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Store each coefficient as a lot-level scalar (broadcast to all rows)
        for d in range(n_basis):
            latent[lot_mask, d] = float(coeffs[d])

    cols = {f"latent_state_{d + 1}": latent[:, d] for d in range(n_dims)}
    return pd.DataFrame(cols, index=df_meta.index)


# ============================================================
# Core
# ============================================================

def infer_one_dataset(
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
    temporal_lot_window_k: int = DEFAULT_TEMPORAL_LOT_WINDOW_K,
    residual_pca_components: int = DEFAULT_RESIDUAL_PCA_COMPONENTS,
    learn_lot_state: bool = DEFAULT_LEARN_LOT_STATE,
    lot_state_dims: int = DEFAULT_LOT_STATE_DIMS,
) -> dict | None:
    required = [target_col, slot_col, time_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ⚠️ skip {dataset_name}: missing {missing}")
        return None

    t_sort0 = time.time()
    df = df.sort_values(time_col, ascending=True).reset_index(drop=True)
    t_sort = time.time() - t_sort0

    if lot_col in df.columns:
        lot_ids = df[lot_col].astype(str)
    elif wafer_id_col in df.columns:
        lot_ids = df[wafer_id_col].astype(str).str[:-2]
        df[lot_col] = lot_ids
    else:
        print(f"  ⚠️ skip {dataset_name}: need '{lot_col}' or '{wafer_id_col}' for compensation")
        return None

    n_total = len(df)
    if n_total < 50:
        print(f"  ⚠️ skip {dataset_name}: too small n={n_total}")
        return None

    _train_end = int(n_total * train_ratio)
    val_end = int(n_total * val_ratio)

    # ── Build slot/reference-MET features (no FDC data used) ─────────────────
    t_prep0 = time.time()
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
        df,
        target_col=target_col,
        slot_col=slot_col,
        lot_col=lot_col,
        reference_slot_ids=reference_slot_ids,
    )
    X_raw = append_global_reference_features(
        df,
        X_raw,
        target_col=target_col,
        slot_col=slot_col,
        lot_col=lot_col,
        reference_slot_ids=reference_slot_ids,
        global_ref_model=global_ref_model,
    )

    # Approach 1 (temporal drift): append cross-lot temporal drift features
    if temporal_lot_window_k > 0:
        X_temporal = build_temporal_lot_features(
            df,
            time_col=time_col,
            lot_col=lot_col,
            target_col=target_col,
            slot_col=slot_col,
            reference_slot_ids=reference_slot_ids,
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
    t_prep = time.time() - t_prep0

    slots = df[slot_col].to_numpy()
    test_is_ref = np.isin(slots[val_end:], reference_slot_ids)
    test_is_nonref = ~test_is_ref
    if test_is_nonref.sum() == 0:
        print(f"  ⚠️ skip {dataset_name}: no non-ref rows in test")
        return None

    print(
        f"  slot/ref features: {fs_info['raw_features']} -> "
        f"miss={fs_info['after_missing_filter']} -> "
        f"var={fs_info['after_variance_filter']} -> "
        f"final={fs_info['after_score_filter']}"
    )

    t_fit0 = time.time()
    model = create_model(
        model_path=model_path,
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        average_before_softmax=average_before_softmax,
        poly_features=poly_features,
        subsample_samples=subsample_samples,
    )
    model.fit(X_selected.iloc[:val_end], y[:val_end])
    t_fit = time.time() - t_fit0

    # Approach 2 (learnable latent state): per-lot state vector, optional two-stage refinement ──
    if learn_lot_state and lot_state_dims > 0 and len(reference_slot_ids) > 0:
        t_latent0 = time.time()
        df_meta_full = pd.DataFrame(
            {lot_col: lot_ids.to_numpy(), slot_col: slots},
            index=df.index,
        )
        latent_df = fit_lot_latent_states(
            model=model,
            X_selected=X_selected,
            y=y,
            df_meta=df_meta_full,
            lot_col=lot_col,
            slot_col=slot_col,
            reference_slot_ids=reference_slot_ids,
            n_dims=lot_state_dims,
        )
        # Augment feature matrix with latent state columns (always included; bypass
        # feature selection since they are the target signal).
        X_selected = pd.concat(
            [X_selected, latent_df.astype(np.float32)], axis=1
        )
        # Refit the model on the augmented training features
        del model
        force_cleanup(light=True)
        model = create_model(
            model_path=model_path,
            n_estimators=n_estimators,
            softmax_temperature=softmax_temperature,
            average_before_softmax=average_before_softmax,
            poly_features=poly_features,
            subsample_samples=subsample_samples,
        )
        model.fit(X_selected.iloc[:val_end], y[:val_end])
        t_latent = time.time() - t_latent0
        print(f"  latent-state fit: dims={lot_state_dims} t={t_latent:.2f}s")

    # ── Probabilistic prediction (mean + CI quantiles) ─────────────────────────
    t_pred0 = time.time()
    ci_quantiles = [ci_quantile_lower, ci_quantile_upper]
    y_pred_raw, quantile_arrays = predict_maybe_batched_with_quantiles(
        model, X_selected.iloc[val_end:], batch_size=predict_batch_size, ci_quantiles=ci_quantiles
    )
    q_lower_raw, q_upper_raw = quantile_arrays[0], quantile_arrays[1]
    t_pred = time.time() - t_pred0

    infer_time = t_fit + t_pred

    del model
    force_cleanup(light=True)

    # ── Residual compensation (applied to mean predictions and CI bounds) ──────
    t_comp0 = time.time()
    meta_test = pd.DataFrame(
        {
            lot_col: lot_ids.iloc[val_end:].to_numpy(),
            slot_col: slots[val_end:],
        }
    )
    y_test = y[val_end:]
    y_pred = apply_residual_compensation(
        df_meta=meta_test,
        y_true=y_test,
        y_pred=y_pred_raw,
        lot_col=lot_col,
        slot_col=slot_col,
        reference_slot_ids=reference_slot_ids,
    )
    # The per-sample bias shift from residual compensation must be applied
    # consistently to the CI bounds so that the interval remains centred on
    # the compensated mean prediction.
    bias_shift = y_pred - y_pred_raw
    q_lower = q_lower_raw + bias_shift
    q_upper = q_upper_raw + bias_shift
    t_comp = time.time() - t_comp0

    # ── Evaluate (non-ref only) with probability interval metrics ─────────────
    test_is_nonref_mask = ~test_is_ref
    metrics = eval_metrics_prob(
        y_true=y_test[test_is_nonref_mask],
        y_pred=y_pred[test_is_nonref_mask],
        q_lower=q_lower[test_is_nonref_mask],
        q_upper=q_upper[test_is_nonref_mask],
        conf_width_thresholds=conf_width_thresholds,
    )

    t_plot0 = time.time()
    safe = dataset_name.replace("/", "_").replace(" ", "_").replace(".", "_")
    plot_path = os.path.join(output_dir, f"{safe}_infer_timeseries.png")
    ci_quantile_label = f"CI[{ci_quantile_lower:.0%},{ci_quantile_upper:.0%}]"
    plot_pred_true_timeseries(
        y_test=y_test,
        y_pred=y_pred,
        test_is_ref=test_is_ref,
        title=(
            f"{dataset_name} | COMP Non-ref MAE={metrics['mae']:.4f} R²={metrics['r2']:.4f} "
            f"Acc@0.5={metrics['acc05']:.1f}% Acc@1.0={metrics['acc10']:.1f}% "
            f"CI-width={metrics['ci_width_mean']:.3f} {ci_quantile_label}"
        ),
        out_path=plot_path,
        ylabel=target_col,
        q_lower=q_lower,
        q_upper=q_upper,
    )
    t_plot = time.time() - t_plot0

    print(
        f"  timing: sort={t_sort:.2f}s prep={t_prep:.2f}s "
        f"fit={t_fit:.2f}s pred={t_pred:.2f}s comp={t_comp:.2f}s plot={t_plot:.2f}s"
    )

    return {
        "dataset": dataset_name,
        "n_rows": int(n_total),
        "n_features_raw": int(fs_info["raw_features"]),
        "n_features_used": int(len(selected_cols)),
        "n_test": int(n_total - val_end),
        "n_test_nonref": int(test_is_nonref.sum()),
        "time_sec": float(infer_time),
        "metrics": metrics,
        "plot": plot_path,
    }


# ============================================================
# CLI
# ============================================================

def _parse_reference_slot_ids(s: str) -> list[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fast TabPFN baseline inference with aggressive feature pruning and per-lot residual compensation."
    )

    p.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH, help="Input file or folder (csv/parquet).")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output folder for plots/results.")

    p.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    p.add_argument("--time-col", type=str, default=DEFAULT_TIME_COL)
    p.add_argument("--slot-col", type=str, default=DEFAULT_SLOT_COL)
    p.add_argument("--lot-col", type=str, default=DEFAULT_LOT_COL)
    p.add_argument("--wafer-id-col", type=str, default=DEFAULT_WAFER_ID_COL)

    p.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    p.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)

    p.add_argument(
        "--reference-slot-ids",
        type=str,
        default=DEFAULT_REFERENCE_SLOT_IDS,
        help='Comma-separated slot ids, e.g. "2,3,4,5,12,13,20,21,22,23".',
    )

    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    p.add_argument("--softmax-temperature", type=float, default=DEFAULT_SOFTMAX_TEMPERATURE)
    p.add_argument(
        "--average-before-softmax",
        action="store_true",
        default=DEFAULT_AVERAGE_BEFORE_SOFTMAX,
        help="If set, enable average_before_softmax.",
    )
    p.add_argument("--poly-features", type=int, default=DEFAULT_POLY_FEATURES)
    p.add_argument("--subsample-samples", type=int, default=DEFAULT_SUBSAMPLE_SAMPLES)
    p.add_argument(
        "--predict-batch-size",
        type=int,
        default=DEFAULT_PREDICT_BATCH_SIZE,
        help="0 means predict all test rows at once.",
    )

    p.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    p.add_argument("--max-missing-ratio", type=float, default=DEFAULT_MAX_MISSING_RATIO)
    p.add_argument("--min-variance", type=float, default=DEFAULT_MIN_VARIANCE)

    # ── Learnable prior-state features ────────────────────────────────────────
    p.add_argument(
        "--temporal-lot-window-k",
        type=int,
        default=DEFAULT_TEMPORAL_LOT_WINDOW_K,
        help=(
            "Approach 1 (temporal drift): window size for cross-lot drift features. "
            "Uses the K most recent preceding lots' reference-MET statistics as features. "
            "Set to 0 to disable."
        ),
    )
    p.add_argument(
        "--residual-pca-components",
        type=int,
        default=DEFAULT_RESIDUAL_PCA_COMPONENTS,
        help=(
            "Approach 3 (residual PCA): number of residual PCA components. After the primary "
            "global-reference-template PCA, a second SVD is applied to the unexplained residuals "
            "and the top-N components are added as features. Set to 0 to disable."
        ),
    )
    p.add_argument(
        "--learn-lot-state",
        action="store_true",
        default=DEFAULT_LEARN_LOT_STATE,
        help=(
            "Approach 2 (learnable latent state): enable per-lot latent state vectors. "
            "A K-dim state is fitted per lot from reference-wafer residuals and appended "
            "as features; the model is then refit on the augmented feature set."
        ),
    )
    p.add_argument(
        "--lot-state-dims",
        type=int,
        default=DEFAULT_LOT_STATE_DIMS,
        help="Approach 2 (learnable latent state): dimensionality K of the per-lot state vector (requires --learn-lot-state).",
    )

    p.add_argument(
        "--ci-quantile-lower",
        type=float,
        default=DEFAULT_CI_QUANTILE_LOWER,
        help=(
            "Lower quantile for prediction interval (must be in [0, 1]). "
            "Paired with --ci-quantile-upper: e.g. 0.1 + 0.9 gives an 80%% PI."
        ),
    )
    p.add_argument(
        "--ci-quantile-upper",
        type=float,
        default=DEFAULT_CI_QUANTILE_UPPER,
        help=(
            "Upper quantile for prediction interval (must be in [0, 1] and > --ci-quantile-lower). "
            "Paired with --ci-quantile-lower: e.g. 0.1 + 0.9 gives an 80%% PI."
        ),
    )
    p.add_argument(
        "--conf-width-thresholds",
        type=str,
        default=DEFAULT_CONF_WIDTH_THRESHOLDS,
        help='Comma-separated CI-width thresholds for high-confidence evaluation, e.g. "1.0,2.0,3.0".',
    )

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    reference_slot_ids = _parse_reference_slot_ids(args.reference_slot_ids)
    conf_width_thresholds = [float(x.strip()) for x in args.conf_width_thresholds.split(",") if x.strip()]

    # Validate CI quantile arguments
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

    files = discover_files(args.data_path)
    print(f"Found {len(files)} file(s). OUTPUT_DIR={args.output_dir}")
    print(
        f"CI interval: [{args.ci_quantile_lower:.0%}, {args.ci_quantile_upper:.0%}] | "
        f"conf-width thresholds: {conf_width_thresholds}"
    )

    all_results = []
    t_all = time.time()

    for i, fp in enumerate(files):
        name = os.path.basename(fp)
        print(f"\n[{i+1}/{len(files)}] {name}")
        df = load_single_file(fp)
        print(f"  shape={df.shape}")

        try:
            res = infer_one_dataset(
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
                temporal_lot_window_k=args.temporal_lot_window_k,
                residual_pca_components=args.residual_pca_components,
                learn_lot_state=args.learn_lot_state,
                lot_state_dims=args.lot_state_dims,
            )
            if res is not None:
                all_results.append(res)
                m = res["metrics"]
                print(
                    f"  COMP Non-ref: MAE={m['mae']:.4f} R²={m['r2']:.4f} "
                    f"Acc@0.5={m['acc05']:.1f}% Acc@1.0={m['acc10']:.1f}% "
                    f"| CI-width mean={m['ci_width_mean']:.3f} median={m['ci_width_median']:.3f} "
                    f"| empirical-coverage={m['ci_empirical_coverage_pct']:.1f}%"
                )
                for thr in conf_width_thresholds:
                    key = f"ci_thr{thr:.1f}"
                    cov = m.get(f"{key}_coverage_pct", float("nan"))
                    thr_mae = m.get(f"{key}_mae", float("nan"))
                    thr_acc05 = m.get(f"{key}_acc05", float("nan"))
                    thr_acc10 = m.get(f"{key}_acc10", float("nan"))
                    print(
                        f"    CI-width≤{thr:.1f}: coverage={cov:.1f}% "
                        f"MAE={thr_mae:.4f} Acc@0.5={thr_acc05:.1f}% Acc@1.0={thr_acc10:.1f}%"
                    )
                print(
                    f"  | time={res['time_sec']:.1f}s "
                    f"| features={res['n_features_raw']}->{res['n_features_used']}"
                )
                print(f"  plot={res['plot']}")
        except Exception as e:
            print(f"  ❌ failed: {type(e).__name__}: {e}")
        finally:
            del df
            force_cleanup(light=True)

    print(f"\nDone. success={len(all_results)}/{len(files)} total_time={time.time()-t_all:.1f}s")

    if all_results:
        avg_mae = float(np.mean([r["metrics"]["mae"] for r in all_results]))
        avg_acc05 = float(np.mean([r["metrics"]["acc05"] for r in all_results]))
        avg_acc10 = float(np.mean([r["metrics"]["acc10"] for r in all_results]))
        avg_ci_width = float(np.mean([r["metrics"]["ci_width_mean"] for r in all_results]))
        avg_ci_cov = float(np.mean([r["metrics"]["ci_empirical_coverage_pct"] for r in all_results]))
        print(
            f"AVG COMP Non-ref MAE={avg_mae:.4f} | "
            f"AVG Acc@0.5={avg_acc05:.1f}% | AVG Acc@1.0={avg_acc10:.1f}% | "
            f"AVG CI-width={avg_ci_width:.3f} | AVG empirical-coverage={avg_ci_cov:.1f}%"
        )
        for thr in conf_width_thresholds:
            key = f"ci_thr{thr:.1f}"
            valid = [r["metrics"] for r in all_results if not np.isnan(r["metrics"].get(f"{key}_mae", float("nan")))]
            if valid:
                avg_thr_cov = float(np.mean([m[f"{key}_coverage_pct"] for m in valid]))
                avg_thr_mae = float(np.mean([m[f"{key}_mae"] for m in valid]))
                avg_thr_acc05 = float(np.mean([m[f"{key}_acc05"] for m in valid]))
                print(
                    f"  AVG CI-width≤{thr:.1f}: coverage={avg_thr_cov:.1f}% "
                    f"MAE={avg_thr_mae:.4f} Acc@0.5={avg_thr_acc05:.1f}%"
                )


if __name__ == "__main__":
    main()
    print("Process finished, exiting now...", flush=True)
    os._exit(0)
