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
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.feature_selection import f_regression

from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Degrees of freedom")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'.*")


# ============================================================
# Route definitions
# ============================================================

ROUTE_INFO = {
    "C_loop_count": {
        "principle": "Directly regress the final ordinal loop_count and round/clip the continuous loop score.",
        "implementation_diff": "Target is loop_count. Prediction is pred_loop_cont -> round/clip. Confidence uses distance to 2.5/3.5/... loop boundaries.",
    },
    "F_delta_run_trend": {
        "principle": "Predict reference-relative delta_run using FDC plus lot-level reference slot trend features.",
        "implementation_diff": "Target is run_value - lot_ref_run_median. Features include reference summary, slot_delta_prior, and lot-internal trend features. Prediction is lot_ref_run_median + pred_delta_run -> loop.",
    },
}


# ============================================================
# Defaults
# ============================================================

DEFAULT_DATA_PATH = "/ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB06_CHA1_1011_1229.parquet"
DEFAULT_DATA_PATH = "/ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/"
DEFAULT_OUTPUT_DIR = "./results/two_routes_calibrated_high_conf"

DEFAULT_TARGET_COL = "met"
DEFAULT_TIME_COL = "start_time"
DEFAULT_SLOT_COL = "slot_id"
DEFAULT_LOT_COL = "lot_id"
DEFAULT_WAFER_ID_COL = "wafer_id"

DEFAULT_REFERENCE_SLOT_IDS = "2,3,4,5,12,13,20,21,22,23"

DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_VAL_RATIO = 0.95

DEFAULT_MODEL_PATH = "/ossfs/workspace/xrfm/TabPFN-main/models/tabpfn-v2.5-regressor-v2.5_default.ckpt"

DEFAULT_N_ESTIMATORS = 4
DEFAULT_SOFTMAX_TEMPERATURE = 0.9
DEFAULT_AVERAGE_BEFORE_SOFTMAX = True
DEFAULT_POLY_FEATURES = 1
DEFAULT_SUBSAMPLE_SAMPLES = 2048
DEFAULT_PREDICT_BATCH_SIZE = 0

DEFAULT_MAX_FEATURES = 120
DEFAULT_MAX_MISSING_RATIO = 0.60
DEFAULT_MIN_VARIANCE = 1e-10

CLASS_LABELS = np.arange(2, 10, dtype=int)
RUN_VALUE_BOUNDS = np.array(
    [0.0, 19.5, 26.2, 33.0, 39.8, 46.5, 53.5, 60.1, 100.0],
    dtype=np.float32,
)
ROUND_BOUNDARIES = np.arange(
    CLASS_LABELS[0] + 0.5,
    CLASS_LABELS[-1] + 0.5,
    1.0,
    dtype=np.float32,
)

DEFAULT_DIFF_PENALTY_POWER = 2.0

DEFAULT_TARGET_HC_ACC = 95.0
DEFAULT_MAX_HC_SEVERE = 0.0
DEFAULT_MIN_CAL_HC_N = 10
DEFAULT_MIN_TEST_HC_N_WARN = 10

DEFAULT_SLOT_DELTA_MIN_COUNT = 3
DEFAULT_SLOT_DELTA_FALLBACK = 0.0


# ============================================================
# Logging / IO
# ============================================================

def log_section(title: str) -> None:
    print("\n" + "=" * 130)
    print(title)
    print("=" * 130)


def log_subsection(title: str) -> None:
    print("\n" + "-" * 130)
    print(title)
    print("-" * 130)


def print_df(df: pd.DataFrame, *, max_rows: int = 80) -> None:
    if df is None or len(df) == 0:
        print("  <empty>")
        return

    old_max_rows = pd.get_option("display.max_rows")
    old_width = pd.get_option("display.width")
    old_float = pd.get_option("display.float_format")

    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.width", 320)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    print(df.head(max_rows).to_string(index=False))

    pd.set_option("display.max_rows", old_max_rows)
    pd.set_option("display.width", old_width)
    pd.set_option("display.float_format", old_float)


def force_cleanup(light: bool = True) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if not light:
            torch.cuda.synchronize()


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


def parse_reference_slot_ids(s: str) -> list[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ============================================================
# Label transforms
# ============================================================

def met_to_run_value(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    return (81.0 - y - 0.3127) / 0.1313 - 6.0


def run_value_to_met(run_value: np.ndarray) -> np.ndarray:
    run_value = np.asarray(run_value, dtype=np.float32)
    return 79.8995 - 0.1313 * run_value


def run_value_to_final_y(run_value: np.ndarray, *, out_of_range: str = "clip") -> np.ndarray:
    rv = np.asarray(run_value, dtype=np.float32)
    idx = np.searchsorted(RUN_VALUE_BOUNDS[1:-1], rv, side="right")
    final_y = CLASS_LABELS[np.clip(idx, 0, len(CLASS_LABELS) - 1)].astype(np.float32)

    in_range = (rv >= RUN_VALUE_BOUNDS[0]) & (rv < RUN_VALUE_BOUNDS[-1])

    if out_of_range == "clip":
        return final_y

    if out_of_range == "nan":
        final_y = final_y.astype(np.float32)
        final_y[~in_range] = np.nan
        return final_y

    if out_of_range == "error":
        if not np.all(in_range):
            bad = int((~in_range).sum())
            raise ValueError(f"Found {bad} run_value values outside [0, 100).")
        return final_y

    raise ValueError(f"Unsupported out_of_range policy: {out_of_range}")


def met_to_final_y(y: np.ndarray, *, out_of_range: str = "clip") -> tuple[np.ndarray, np.ndarray, int]:
    run_value = met_to_run_value(y)
    final_y = run_value_to_final_y(run_value, out_of_range=out_of_range)
    n_out = int(((run_value < RUN_VALUE_BOUNDS[0]) | (run_value >= RUN_VALUE_BOUNDS[-1])).sum())
    return final_y, run_value, n_out


def round_clip_final_y(y_pred_cont: np.ndarray) -> np.ndarray:
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float32)
    return np.clip(np.rint(y_pred_cont), CLASS_LABELS[0], CLASS_LABELS[-1]).astype(int)


def distance_to_run_boundary_nm(pred_run: np.ndarray) -> np.ndarray:
    pred_run = np.asarray(pred_run, dtype=np.float32).reshape(-1)
    internal_bounds = RUN_VALUE_BOUNDS[1:-1]
    dist_run = np.min(
        np.abs(pred_run.reshape(-1, 1) - internal_bounds.reshape(1, -1)),
        axis=1,
    )
    return dist_run * 0.1313


def distance_to_loop_round_boundary(pred_loop_cont: np.ndarray) -> np.ndarray:
    pred_loop_cont = np.asarray(pred_loop_cont, dtype=np.float32).reshape(-1)
    return np.min(
        np.abs(pred_loop_cont.reshape(-1, 1) - ROUND_BOUNDARIES.reshape(1, -1)),
        axis=1,
    )


# ============================================================
# Metrics
# ============================================================

def safe_balanced_accuracy(y_true_cls: np.ndarray, y_pred_cls: np.ndarray) -> float:
    y_true_cls = np.asarray(y_true_cls, dtype=int)
    y_pred_cls = np.asarray(y_pred_cls, dtype=int)

    if len(np.unique(y_true_cls)) <= 1:
        return float(np.mean(y_pred_cls == y_true_cls) * 100.0)

    return float(balanced_accuracy_score(y_true_cls, y_pred_cls) * 100.0)


def eval_loop_metrics(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    *,
    penalty_power: float,
) -> dict:
    y_true_cls = np.asarray(y_true_cls, dtype=int)
    y_pred_cls = np.asarray(y_pred_cls, dtype=int)

    if len(y_true_cls) == 0:
        raise ValueError("Empty y_true for metrics")

    diff = y_pred_cls - y_true_cls
    abs_diff = np.abs(diff)

    max_diff = int(CLASS_LABELS[-1] - CLASS_LABELS[0])
    weighted_penalty = float(np.mean(abs_diff.astype(np.float32) ** penalty_power))
    worst_penalty = float(max_diff ** penalty_power)
    control_score = float(max(0.0, 100.0 * (1.0 - weighted_penalty / worst_penalty)))

    return {
        "n": int(len(y_true_cls)),
        "accuracy": float(accuracy_score(y_true_cls, y_pred_cls) * 100.0),
        "balanced_accuracy": safe_balanced_accuracy(y_true_cls, y_pred_cls),
        "macro_f1": float(
            f1_score(
                y_true_cls,
                y_pred_cls,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            ) * 100.0
        ),
        "within_1": float(np.mean(abs_diff <= 1) * 100.0),
        "within_2": float(np.mean(abs_diff <= 2) * 100.0),
        "severe_ge2": float(np.mean(abs_diff >= 2) * 100.0),
        "extreme_ge3": float(np.mean(abs_diff >= 3) * 100.0),
        "mae_class": float(mean_absolute_error(y_true_cls, y_pred_cls)),
        "rmse_class": float(np.sqrt(mean_squared_error(y_true_cls, y_pred_cls))),
        "mean_signed_diff": float(np.mean(diff)),
        "weighted_penalty": weighted_penalty,
        "control_score": control_score,
    }


def safe_eval_loop(y_true: np.ndarray, y_pred: np.ndarray, *, penalty_power: float) -> dict | None:
    if len(y_true) == 0:
        return None
    return eval_loop_metrics(y_true, y_pred, penalty_power=penalty_power)


def print_metric_block(title: str, metrics: dict | None) -> None:
    print(f"\n  [{title}]")
    if metrics is None:
        print("    <empty>")
        return

    print(
        f"    n={metrics['n']} | "
        f"Acc={metrics['accuracy']:.2f}% | "
        f"Within1={metrics['within_1']:.2f}% | "
        f"Severe(|d|>=2)={metrics['severe_ge2']:.2f}% | "
        f"MAE={metrics['mae_class']:.4f} | "
        f"RMSE={metrics['rmse_class']:.4f} | "
        f"Penalty={metrics['weighted_penalty']:.4f} | "
        f"MeanDiff={metrics['mean_signed_diff']:.4f}"
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    if valid.sum() == 0:
        return {
            f"{prefix}_mae": np.nan,
            f"{prefix}_rmse": np.nan,
            f"{prefix}_mean_err": np.nan,
            f"{prefix}_p50_abs_err": np.nan,
            f"{prefix}_p90_abs_err": np.nan,
        }

    err = y_pred[valid] - y_true[valid]
    return {
        f"{prefix}_mae": float(np.mean(np.abs(err))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(err ** 2))),
        f"{prefix}_mean_err": float(np.mean(err)),
        f"{prefix}_p50_abs_err": float(np.percentile(np.abs(err), 50)),
        f"{prefix}_p90_abs_err": float(np.percentile(np.abs(err), 90)),
    }


def distribution_table(name: str, values: np.ndarray, labels: np.ndarray = CLASS_LABELS) -> pd.DataFrame:
    values = np.asarray(values)
    rows = []
    n = len(values)

    for lab in labels:
        c = int(np.sum(values == lab))
        rows.append({name: int(lab), "count": c, "ratio_pct": 100.0 * c / max(n, 1)})

    return pd.DataFrame(rows)


# ============================================================
# Reference trend features
# ============================================================

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 2:
        return 0.0

    xv = x[valid]
    yv = y[valid]

    if np.nanstd(xv) < 1e-8 or np.nanstd(yv) < 1e-8:
        return 0.0

    return float(np.corrcoef(xv, yv)[0, 1])


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 2:
        if valid.sum() == 1:
            return 0.0, float(y[valid][0])
        return 0.0, np.nan

    xv = x[valid]
    yv = y[valid]

    if np.nanstd(xv) < 1e-8:
        return 0.0, float(np.nanmean(yv))

    slope, intercept = np.polyfit(xv, yv, deg=1)
    return float(slope), float(intercept)


def compute_ref_trend_features_for_window(
    *,
    lots: np.ndarray,
    slots: np.ndarray,
    run_value: np.ndarray,
    loop_value: np.ndarray,
    reference_slot_ids: list[int],
) -> pd.DataFrame:
    lots = np.asarray(lots)
    slots = np.asarray(slots).astype(float)
    run_value = np.asarray(run_value, dtype=np.float32)
    loop_value = np.asarray(loop_value, dtype=np.float32)

    is_ref = np.isin(slots.astype(int), reference_slot_ids)

    out = pd.DataFrame(index=np.arange(len(lots)))

    cols = [
        "ref_run_median",
        "ref_run_mean",
        "ref_run_std",
        "ref_run_min",
        "ref_run_max",
        "ref_run_range",
        "ref_loop_median",
        "ref_loop_min",
        "ref_loop_max",
        "n_ref",
        "ref_slot_min",
        "ref_slot_max",
        "ref_slot_median",
        "ref_slot_range",
        "ref_trend_slope",
        "ref_trend_intercept",
        "ref_trend_corr",
        "ref_trend_abs_corr",
        "ref_trend_pred_run",
        "slot_numeric",
        "slot_rank_in_lot",
        "slot_rank_pct_in_lot",
        "slot_centered_in_lot",
        "abs_slot_centered_in_lot",
        "slot_minus_ref_median",
        "slot_abs_minus_ref_median",
        "slot_abs_dist_nearest_ref",
        "slot_between_ref_min_max",
    ]

    for c in cols:
        out[c] = np.nan

    out["n_ref"] = 0
    out["slot_numeric"] = slots
    out["slot_between_ref_min_max"] = 0.0

    for lot in np.unique(lots):
        lot_mask = lots == lot
        idx = np.where(lot_mask)[0]
        lot_slots = slots[lot_mask]

        order = np.argsort(lot_slots)
        ranks = np.empty(len(lot_slots), dtype=np.float32)
        ranks[order] = np.arange(len(lot_slots), dtype=np.float32)

        denom = max(len(lot_slots) - 1, 1)
        rank_pct = ranks / denom

        out.loc[idx, "slot_rank_in_lot"] = ranks
        out.loc[idx, "slot_rank_pct_in_lot"] = rank_pct
        out.loc[idx, "slot_centered_in_lot"] = rank_pct - 0.5
        out.loc[idx, "abs_slot_centered_in_lot"] = np.abs(rank_pct - 0.5)

        ref_mask = lot_mask & is_ref
        n_ref = int(ref_mask.sum())
        out.loc[idx, "n_ref"] = n_ref

        if n_ref == 0:
            continue

        ref_slots = slots[ref_mask]
        ref_run = run_value[ref_mask]
        ref_loop = loop_value[ref_mask]

        ref_run_med = float(np.nanmedian(ref_run))
        ref_run_mean = float(np.nanmean(ref_run))
        ref_run_std = float(np.nanstd(ref_run))
        ref_run_min = float(np.nanmin(ref_run))
        ref_run_max = float(np.nanmax(ref_run))

        ref_slot_min = float(np.nanmin(ref_slots))
        ref_slot_max = float(np.nanmax(ref_slots))
        ref_slot_med = float(np.nanmedian(ref_slots))

        slope, intercept = fit_line(ref_slots, ref_run)
        corr = safe_corr(ref_slots, ref_run)

        out.loc[idx, "ref_run_median"] = ref_run_med
        out.loc[idx, "ref_run_mean"] = ref_run_mean
        out.loc[idx, "ref_run_std"] = ref_run_std
        out.loc[idx, "ref_run_min"] = ref_run_min
        out.loc[idx, "ref_run_max"] = ref_run_max
        out.loc[idx, "ref_run_range"] = ref_run_max - ref_run_min

        out.loc[idx, "ref_loop_median"] = float(np.nanmedian(ref_loop))
        out.loc[idx, "ref_loop_min"] = float(np.nanmin(ref_loop))
        out.loc[idx, "ref_loop_max"] = float(np.nanmax(ref_loop))

        out.loc[idx, "ref_slot_min"] = ref_slot_min
        out.loc[idx, "ref_slot_max"] = ref_slot_max
        out.loc[idx, "ref_slot_median"] = ref_slot_med
        out.loc[idx, "ref_slot_range"] = ref_slot_max - ref_slot_min

        out.loc[idx, "ref_trend_slope"] = slope
        out.loc[idx, "ref_trend_intercept"] = intercept
        out.loc[idx, "ref_trend_corr"] = corr
        out.loc[idx, "ref_trend_abs_corr"] = abs(corr)
        out.loc[idx, "ref_trend_pred_run"] = slope * lot_slots + intercept

        out.loc[idx, "slot_minus_ref_median"] = lot_slots - ref_slot_med
        out.loc[idx, "slot_abs_minus_ref_median"] = np.abs(lot_slots - ref_slot_med)

        dist_to_refs = np.min(
            np.abs(lot_slots.reshape(-1, 1) - ref_slots.reshape(1, -1)),
            axis=1,
        )
        out.loc[idx, "slot_abs_dist_nearest_ref"] = dist_to_refs
        out.loc[idx, "slot_between_ref_min_max"] = (
            (lot_slots >= ref_slot_min) & (lot_slots <= ref_slot_max)
        ).astype(float)

    return out


def build_slot_delta_prior(
    *,
    train_slots: np.ndarray,
    train_run_value: np.ndarray,
    train_ref_features: pd.DataFrame,
    reference_slot_ids: list[int],
    min_count: int,
    fallback: float,
) -> tuple[dict, pd.DataFrame, float]:
    is_ref = np.isin(train_slots.astype(int), reference_slot_ids)
    ref_med = train_ref_features["ref_run_median"].to_numpy(dtype=np.float32)

    valid = (~is_ref) & np.isfinite(ref_med) & np.isfinite(train_run_value)

    if valid.sum() == 0:
        return {}, pd.DataFrame(), fallback

    df_delta = pd.DataFrame(
        {
            "slot_id": train_slots[valid].astype(int),
            "delta_run": train_run_value[valid] - ref_med[valid],
        }
    )

    global_median = float(np.nanmedian(df_delta["delta_run"].to_numpy()))

    if not np.isfinite(global_median):
        global_median = fallback

    rows = []
    priors = {}

    for slot, sub in df_delta.groupby("slot_id"):
        arr = sub["delta_run"].to_numpy(dtype=np.float32)
        n = int(len(arr))
        med = float(np.nanmedian(arr))
        mean = float(np.nanmean(arr))
        p25 = float(np.nanpercentile(arr, 25))
        p75 = float(np.nanpercentile(arr, 75))
        iqr = p75 - p25

        used = med if n >= min_count else global_median
        priors[int(slot)] = used

        rows.append(
            {
                "slot_id": int(slot),
                "n": n,
                "delta_run_median": med,
                "delta_run_mean": mean,
                "delta_run_p25": p25,
                "delta_run_p75": p75,
                "delta_run_iqr": iqr,
                "used_delta_run": used,
                "used_global_fallback": n < min_count,
            }
        )

    prior_df = pd.DataFrame(rows).sort_values("slot_id").reset_index(drop=True)
    return priors, prior_df, global_median


def slot_delta_for_slots(slots: np.ndarray, prior: dict, fallback_delta: float) -> np.ndarray:
    return np.array([float(prior.get(int(s), fallback_delta)) for s in slots], dtype=np.float32)


def slot_prior_iqr_for_slots(slots: np.ndarray, prior_df: pd.DataFrame) -> np.ndarray:
    if prior_df is None or len(prior_df) == 0:
        return np.full(len(slots), np.nan, dtype=np.float32)

    mp = {int(r["slot_id"]): float(r["delta_run_iqr"]) for _, r in prior_df.iterrows()}
    return np.array([mp.get(int(s), np.nan) for s in slots], dtype=np.float32)


def slot_delta_rule_loop(
    *,
    ref_run_median: np.ndarray,
    slot_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rule_run = np.asarray(ref_run_median, dtype=np.float32) + np.asarray(slot_delta, dtype=np.float32)
    rule_loop = run_value_to_final_y(rule_run, out_of_range="clip").astype(int)

    invalid = ~np.isfinite(rule_run)
    rule_loop[invalid] = -1

    return rule_run, rule_loop


def trend_rule_loop(
    *,
    trend_pred_run: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rule_run = np.asarray(trend_pred_run, dtype=np.float32)
    rule_loop = run_value_to_final_y(rule_run, out_of_range="clip").astype(int)

    invalid = ~np.isfinite(rule_run)
    rule_loop[invalid] = -1

    return rule_run, rule_loop


# ============================================================
# Compensation / detail / rules
# ============================================================

def apply_residual_compensation_continuous(
    *,
    y_true_cont: np.ndarray,
    y_pred_cont: np.ndarray,
    lots: np.ndarray,
    slots: np.ndarray,
    reference_slot_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_cont = np.asarray(y_true_cont, dtype=np.float32)
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float32)
    lots = np.asarray(lots)
    slots = np.asarray(slots)

    is_ref = np.isin(slots.astype(int), reference_slot_ids)

    compensated = y_pred_cont.copy()
    bias_per_row = np.zeros(len(y_pred_cont), dtype=np.float32)
    n_ref_per_row = np.zeros(len(y_pred_cont), dtype=int)

    for lot in np.unique(lots):
        lot_mask = lots == lot
        ref_mask = lot_mask & is_ref
        nonref_mask = lot_mask & (~is_ref)

        n_ref = int(ref_mask.sum())

        if n_ref == 0:
            continue

        bias = float(np.nanmean(y_true_cont[ref_mask] - y_pred_cont[ref_mask]))

        if not np.isfinite(bias):
            bias = 0.0
        else:
            compensated[nonref_mask] += bias

        bias_per_row[lot_mask] = bias
        n_ref_per_row[lot_mask] = n_ref

    return compensated, bias_per_row, n_ref_per_row


def compute_ref_bias_loop(
    *,
    y_true_loop: np.ndarray,
    y_pred_loop_raw: np.ndarray,
    lots: np.ndarray,
    slots: np.ndarray,
    reference_slot_ids: list[int],
) -> np.ndarray:
    y_true_loop = np.asarray(y_true_loop, dtype=np.float32)
    y_pred_loop_raw = np.asarray(y_pred_loop_raw, dtype=np.float32)
    lots = np.asarray(lots)
    slots = np.asarray(slots)

    is_ref = np.isin(slots.astype(int), reference_slot_ids)
    bias = np.zeros(len(y_true_loop), dtype=np.float32)

    for lot in np.unique(lots):
        lot_mask = lots == lot
        ref_mask = lot_mask & is_ref

        if ref_mask.sum() == 0:
            continue

        b = float(np.nanmean(y_true_loop[ref_mask] - y_pred_loop_raw[ref_mask]))

        if np.isfinite(b):
            bias[lot_mask] = b

    return bias


def build_prediction_detail(
    *,
    model_name: str,
    target_type: str,
    pred_raw: np.ndarray,
    y_true_met: np.ndarray,
    y_true_run: np.ndarray,
    y_true_loop: np.ndarray,
    lots: np.ndarray,
    slots: np.ndarray,
    ref_features: pd.DataFrame,
    slot_rule_run: np.ndarray,
    slot_rule_loop_arr: np.ndarray,
    trend_rule_run: np.ndarray,
    trend_rule_loop_arr: np.ndarray,
    nonref_mask: np.ndarray,
    reference_slot_ids: list[int],
    use_residual_compensation: bool,
) -> pd.DataFrame:
    if target_type == "loop":
        pred_loop_cont_raw = pred_raw

        if use_residual_compensation:
            pred_loop_cont, _, n_ref_native = apply_residual_compensation_continuous(
                y_true_cont=y_true_loop.astype(np.float32),
                y_pred_cont=pred_loop_cont_raw,
                lots=lots,
                slots=slots,
                reference_slot_ids=reference_slot_ids,
            )
        else:
            pred_loop_cont = pred_loop_cont_raw
            n_ref_native = ref_features["n_ref"].to_numpy(dtype=int)

        pred_loop_raw = round_clip_final_y(pred_loop_cont_raw)
        pred_loop = round_clip_final_y(pred_loop_cont)
        pred_run = np.full(len(pred_loop), np.nan, dtype=np.float32)
        boundary_dist = distance_to_loop_round_boundary(pred_loop_cont)
        valid = np.isfinite(pred_loop_cont)

        rule_run_gap_slot = np.where(pred_loop == slot_rule_loop_arr, 0.0, 999.0).astype(np.float32)
        rule_run_gap_trend = np.where(pred_loop == trend_rule_loop_arr, 0.0, 999.0).astype(np.float32)

    elif target_type == "delta_run_trend":
        ref_run = ref_features["ref_run_median"].to_numpy(dtype=np.float32)
        pred_delta = pred_raw
        pred_run_raw = ref_run + pred_delta

        if use_residual_compensation:
            pred_run, _, n_ref_native = apply_residual_compensation_continuous(
                y_true_cont=y_true_run,
                y_pred_cont=pred_run_raw,
                lots=lots,
                slots=slots,
                reference_slot_ids=reference_slot_ids,
            )
        else:
            pred_run = pred_run_raw
            n_ref_native = ref_features["n_ref"].to_numpy(dtype=int)

        pred_loop_raw = run_value_to_final_y(pred_run_raw, out_of_range="clip").astype(int)
        pred_loop = run_value_to_final_y(pred_run, out_of_range="clip").astype(int)
        boundary_dist = distance_to_run_boundary_nm(pred_run)
        valid = np.isfinite(pred_run)

        rule_run_gap_slot = np.abs(pred_run - slot_rule_run)
        rule_run_gap_trend = np.abs(pred_run - trend_rule_run)

    else:
        raise ValueError(f"Unsupported target_type for this two-route script: {target_type}")

    ref_bias_loop = compute_ref_bias_loop(
        y_true_loop=y_true_loop,
        y_pred_loop_raw=pred_loop_raw.astype(np.float32),
        lots=lots,
        slots=slots,
        reference_slot_ids=reference_slot_ids,
    )

    comp_shift = np.abs(pred_loop - pred_loop_raw)

    agree_slot_rule = pred_loop == slot_rule_loop_arr
    agree_trend_rule = pred_loop == trend_rule_loop_arr

    detail = pd.DataFrame(
        {
            "model": model_name,
            "target_type": target_type,
            "lot_id": lots,
            "slot_id": slots,
            "is_nonref": nonref_mask,
            "true_met": y_true_met,
            "true_run": y_true_run,
            "true_loop": y_true_loop,
            "pred_loop": pred_loop,
            "pred_loop_raw": pred_loop_raw,
            "slot_delta_rule_loop": slot_rule_loop_arr,
            "trend_rule_loop": trend_rule_loop_arr,
            "slot_delta_rule_run": slot_rule_run,
            "trend_rule_run": trend_rule_run,
            "boundary_dist": boundary_dist,
            "boundary_kind": "loop_distance" if target_type == "loop" else "run_boundary_nm",
            "ref_bias_loop": ref_bias_loop,
            "abs_ref_bias_loop": np.abs(ref_bias_loop),
            "n_ref": n_ref_native,
            "comp_shift": comp_shift,
            "agree_slot_rule": agree_slot_rule,
            "agree_trend_rule": agree_trend_rule,
            "agree_both": agree_slot_rule & agree_trend_rule,
            "agree_either": agree_slot_rule | agree_trend_rule,
            "rule_run_gap_slot": rule_run_gap_slot,
            "rule_run_gap_trend": rule_run_gap_trend,
            "valid": valid,
            "ref_trend_abs_corr": ref_features["ref_trend_abs_corr"].to_numpy(dtype=np.float32),
            "slot_between_ref": ref_features["slot_between_ref_min_max"].to_numpy(dtype=np.float32),
            "ref_run_std": ref_features["ref_run_std"].to_numpy(dtype=np.float32),
            "slot_delta_iqr": ref_features["slot_delta_iqr"].to_numpy(dtype=np.float32),
        }
    )

    if target_type == "loop":
        detail["pred_run"] = np.nan
        detail["pred_met_from_pred"] = np.nan
    else:
        detail["pred_run"] = pred_run
        detail["pred_met_from_pred"] = run_value_to_met(pred_run)

    detail["diff"] = detail["pred_loop"] - detail["true_loop"]
    detail["abs_diff"] = np.abs(detail["diff"])

    return detail


def apply_rule_mask(df: pd.DataFrame, rule: dict) -> np.ndarray:
    mode = rule["mode"]

    if mode == "slot":
        agree = df["agree_slot_rule"].to_numpy(dtype=bool)
        rule_gap = df["rule_run_gap_slot"].to_numpy(dtype=np.float32)
    elif mode == "trend":
        agree = df["agree_trend_rule"].to_numpy(dtype=bool)
        rule_gap = df["rule_run_gap_trend"].to_numpy(dtype=np.float32)
    elif mode == "both":
        agree = df["agree_slot_rule"].to_numpy(dtype=bool) & df["agree_trend_rule"].to_numpy(dtype=bool)
        rule_gap = np.minimum(
            df["rule_run_gap_slot"].to_numpy(dtype=np.float32),
            df["rule_run_gap_trend"].to_numpy(dtype=np.float32),
        )
    elif mode == "either":
        agree = df["agree_slot_rule"].to_numpy(dtype=bool) | df["agree_trend_rule"].to_numpy(dtype=bool)
        rule_gap = np.minimum(
            df["rule_run_gap_slot"].to_numpy(dtype=np.float32),
            df["rule_run_gap_trend"].to_numpy(dtype=np.float32),
        )
    else:
        raise ValueError(f"Unsupported rule mode: {mode}")

    mask = (
        df["valid"].to_numpy(dtype=bool)
        & (df["boundary_dist"].to_numpy(dtype=np.float32) >= float(rule["boundary_min"]))
        & (df["abs_ref_bias_loop"].to_numpy(dtype=np.float32) <= float(rule["abs_ref_bias_loop_max"]))
        & (df["n_ref"].to_numpy(dtype=int) >= int(rule["hc_min_ref"]))
        & (df["comp_shift"].to_numpy(dtype=np.float32) <= float(rule["hc_max_comp_shift"]))
        & agree
        & (np.nan_to_num(rule_gap, nan=999.0) <= float(rule["rule_run_gap_max"]))
        & (np.nan_to_num(df["ref_trend_abs_corr"].to_numpy(dtype=np.float32), nan=0.0) >= float(rule["ref_trend_abs_corr_min"]))
        & (np.nan_to_num(df["ref_run_std"].to_numpy(dtype=np.float32), nan=999.0) <= float(rule["ref_run_std_max"]))
        & (np.nan_to_num(df["slot_delta_iqr"].to_numpy(dtype=np.float32), nan=999.0) <= float(rule["slot_delta_iqr_max"]))
    )

    if mode in ["trend", "both"]:
        mask = mask & (df["slot_between_ref"].to_numpy(dtype=np.float32) >= 1.0)

    return mask


def evaluate_detail_full(detail: pd.DataFrame, penalty_power: float) -> dict | None:
    mask = detail["is_nonref"].to_numpy(dtype=bool) & detail["valid"].to_numpy(dtype=bool)
    return safe_eval_loop(
        detail.loc[mask, "true_loop"].to_numpy(dtype=int),
        detail.loc[mask, "pred_loop"].to_numpy(dtype=int),
        penalty_power=penalty_power,
    )


def evaluate_detail_rule(detail: pd.DataFrame, rule: dict, penalty_power: float) -> tuple[dict | None, np.ndarray]:
    mask = apply_rule_mask(detail, rule) & detail["is_nonref"].to_numpy(dtype=bool)

    metrics = safe_eval_loop(
        detail.loc[mask, "true_loop"].to_numpy(dtype=int),
        detail.loc[mask, "pred_loop"].to_numpy(dtype=int),
        penalty_power=penalty_power,
    )

    return metrics, mask


# ============================================================
# High-confidence scan / selection
# ============================================================

def high_conf_scan_on_detail(
    *,
    detail: pd.DataFrame,
    model_name: str,
    target_type: str,
    mode: str,
    penalty_power: float,
    min_scan_n: int,
    hc_min_ref: int,
    hc_max_comp_shift: int,
) -> pd.DataFrame:
    is_loop_boundary = target_type == "loop"

    if is_loop_boundary:
        boundary_values = [0.00, 0.20, 0.30, 0.40, 0.45]
    else:
        boundary_values = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]

    bias_values = [0.25, 0.50, 0.75, 1.00, 1.50]
    rule_gap_values = [999.0, 4.0, 3.0, 2.0, 1.5]
    corr_values = [0.0, 0.5, 0.7]
    ref_std_values = [999.0, 4.0, 3.0, 2.0, 1.0]
    slot_iqr_values = [999.0, 4.0, 3.0, 2.0]

    rows = []
    total = int((detail["is_nonref"] & detail["valid"]).sum())

    for bd in boundary_values:
        for cap in bias_values:
            for rgap in rule_gap_values:
                for cmin in corr_values:
                    for rstd in ref_std_values:
                        for siqr in slot_iqr_values:
                            rule = {
                                "model": model_name,
                                "target_type": target_type,
                                "mode": mode,
                                "boundary_min": bd,
                                "abs_ref_bias_loop_max": cap,
                                "rule_run_gap_max": rgap,
                                "ref_trend_abs_corr_min": cmin,
                                "ref_run_std_max": rstd,
                                "slot_delta_iqr_max": siqr,
                                "hc_min_ref": hc_min_ref,
                                "hc_max_comp_shift": hc_max_comp_shift,
                            }

                            mask = apply_rule_mask(detail, rule) & detail["is_nonref"].to_numpy(dtype=bool)
                            n = int(mask.sum())

                            if n < min_scan_n:
                                continue

                            m = eval_loop_metrics(
                                detail.loc[mask, "true_loop"].to_numpy(dtype=int),
                                detail.loc[mask, "pred_loop"].to_numpy(dtype=int),
                                penalty_power=penalty_power,
                            )

                            rows.append(
                                {
                                    **rule,
                                    "n": n,
                                    "coverage_pct": 100.0 * n / max(total, 1),
                                    "accuracy": m["accuracy"],
                                    "within1": m["within_1"],
                                    "severe_ge2": m["severe_ge2"],
                                    "penalty": m["weighted_penalty"],
                                    "mean_diff": m["mean_signed_diff"],
                                }
                            )

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(
            ["accuracy", "coverage_pct", "severe_ge2", "penalty"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)

    return out


def select_rule_from_cal_scan(
    scan_df: pd.DataFrame,
    *,
    target_acc: float,
    max_severe: float,
) -> tuple[dict | None, str]:
    if scan_df is None or len(scan_df) == 0:
        return None, "no_scan_rows"

    ok = scan_df[
        (scan_df["accuracy"] >= target_acc)
        & (scan_df["severe_ge2"] <= max_severe)
        & (scan_df["n"] > 0)
    ].copy()

    if len(ok):
        selected = ok.sort_values(
            ["coverage_pct", "accuracy", "penalty"],
            ascending=[False, False, True],
        ).iloc[0].to_dict()
        return selected, "meet_target"

    selected = scan_df.sort_values(
        ["accuracy", "coverage_pct", "severe_ge2", "penalty"],
        ascending=[False, False, True, True],
    ).iloc[0].to_dict()

    return selected, "fallback_best_accuracy"


# ============================================================
# Feature selection / TabPFN
# ============================================================

def coerce_mixed_columns_for_tabpfn(X: pd.DataFrame) -> pd.DataFrame:
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


def create_model(args: argparse.Namespace) -> TabPFNRegressor:
    return TabPFNRegressor(
        model_path=args.model_path,
        device="cuda",
        n_estimators=args.n_estimators,
        softmax_temperature=args.softmax_temperature,
        average_before_softmax=args.average_before_softmax,
        memory_saving_mode=True,
        ignore_pretraining_limits=True,
        inference_config={
            "SUBSAMPLE_SAMPLES": max(256, int(args.subsample_samples)),
            "POLYNOMIAL_FEATURES": max(1, int(args.poly_features)),
        },
    )


def predict_maybe_batched(model: TabPFNRegressor, X: pd.DataFrame, batch_size: int) -> np.ndarray:
    if batch_size is None or batch_size <= 0 or len(X) <= batch_size:
        return np.asarray(model.predict(X), dtype=np.float32)

    preds = []

    for i in range(0, len(X), batch_size):
        preds.append(np.asarray(model.predict(X.iloc[i:i + batch_size]), dtype=np.float32))

    return np.concatenate(preds)


def fit_predict_tabpfn(
    *,
    model_name: str,
    X_all: pd.DataFrame,
    y_target: np.ndarray,
    train_mask: np.ndarray,
    cal_slice: slice,
    test_slice: slice,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    valid_train = train_mask & np.isfinite(y_target)

    if valid_train.sum() < 30:
        raise ValueError(f"{model_name}: too few valid train rows: {valid_train.sum()}")

    X_train = X_all.loc[valid_train]
    y_train = y_target[valid_train].astype(np.float32)

    X_selected, selected_cols, fs_info = fast_select_features(
        X_train=X_train,
        y_train=y_train,
        X_all=X_all,
        max_features=args.max_features,
        max_missing_ratio=args.max_missing_ratio,
        min_variance=args.min_variance,
    )

    print(
        f"  {model_name}: feature pruning raw={fs_info['raw_features']} -> "
        f"miss={fs_info['after_missing_filter']} -> "
        f"var={fs_info['after_variance_filter']} -> "
        f"final={fs_info['after_score_filter']}"
    )

    model = create_model(args)

    t0 = time.time()
    model.fit(X_selected.loc[valid_train], y_train)
    fit_time = time.time() - t0

    t1 = time.time()
    pred_cal = predict_maybe_batched(model, X_selected.iloc[cal_slice], args.predict_batch_size)
    pred_test = predict_maybe_batched(model, X_selected.iloc[test_slice], args.predict_batch_size)
    pred_time = time.time() - t1

    del model
    force_cleanup(light=True)

    print(f"  {model_name}: fit_time={fit_time:.2f}s pred_time={pred_time:.2f}s")

    return pred_cal, pred_test, selected_cols, fs_info


# ============================================================
# Experiment
# ============================================================

def infer_one_dataset(df: pd.DataFrame, dataset_name: str, args: argparse.Namespace) -> dict | None:
    log_section(f"Dataset: {dataset_name}")

    required = [args.target_col, args.slot_col, args.time_col]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"⚠️ skip {dataset_name}: missing {missing}")
        return None

    t_sort0 = time.time()
    df = df.sort_values(args.time_col, ascending=True).reset_index(drop=True)
    t_sort = time.time() - t_sort0

    if args.lot_col in df.columns:
        lot_ids = df[args.lot_col].astype(str)
    elif args.wafer_id_col in df.columns:
        lot_ids = df[args.wafer_id_col].astype(str).str[:-2]
        df[args.lot_col] = lot_ids
    else:
        print(f"⚠️ skip {dataset_name}: need '{args.lot_col}' or '{args.wafer_id_col}'")
        return None

    y_met = df[args.target_col].astype(float).to_numpy(dtype=np.float32)
    y_loop_float, y_run, n_out_range = met_to_final_y(y_met, out_of_range=args.label_out_of_range)
    y_loop = y_loop_float.astype(int)

    valid_label = np.isfinite(y_loop_float)

    if valid_label.sum() < len(df):
        dropped = int(len(df) - valid_label.sum())
        print(f"drop invalid label rows={dropped}")
        df = df.loc[valid_label].reset_index(drop=True)
        lot_ids = lot_ids.loc[valid_label].reset_index(drop=True)
        y_met = y_met[valid_label]
        y_run = y_run[valid_label]
        y_loop = y_loop[valid_label]

    n_total = len(df)
    train_end = int(n_total * args.train_ratio)
    val_end = int(n_total * args.val_ratio)

    if train_end <= 50 or val_end <= train_end + 20 or n_total <= val_end + 20:
        print(f"⚠️ split too small: n={n_total}, train_end={train_end}, val_end={val_end}")
        return None

    train_slice = slice(0, train_end)
    cal_slice = slice(train_end, val_end)
    test_slice = slice(val_end, n_total)

    lots = lot_ids.to_numpy()
    slots = df[args.slot_col].to_numpy().astype(int)

    print(f"shape={df.shape}, sort_time={t_sort:.3f}s")
    print(f"label out-of-range run_value={n_out_range}/{n_total}, policy={args.label_out_of_range}")
    print(f"split: train=[0,{train_end}), cal=[{train_end},{val_end}), test=[{val_end},{n_total})")
    print(f"split sizes: train={train_end}, cal={val_end-train_end}, test={n_total-val_end}")

    log_subsection("Loop distribution by split")
    dist_train = distribution_table("loop", y_loop[train_slice]).rename(columns={"count": "train_count", "ratio_pct": "train_pct"})
    dist_cal = distribution_table("loop", y_loop[cal_slice]).rename(columns={"count": "cal_count", "ratio_pct": "cal_pct"})
    dist_test = distribution_table("loop", y_loop[test_slice]).rename(columns={"count": "test_count", "ratio_pct": "test_pct"})
    print_df(dist_train.merge(dist_cal, on="loop").merge(dist_test, on="loop"))

    # Split-wise reference features to avoid leakage.
    ref_train = compute_ref_trend_features_for_window(
        lots=lots[train_slice],
        slots=slots[train_slice],
        run_value=y_run[train_slice],
        loop_value=y_loop[train_slice],
        reference_slot_ids=args.reference_slot_ids,
    )
    ref_cal = compute_ref_trend_features_for_window(
        lots=lots[cal_slice],
        slots=slots[cal_slice],
        run_value=y_run[cal_slice],
        loop_value=y_loop[cal_slice],
        reference_slot_ids=args.reference_slot_ids,
    )
    ref_test = compute_ref_trend_features_for_window(
        lots=lots[test_slice],
        slots=slots[test_slice],
        run_value=y_run[test_slice],
        loop_value=y_loop[test_slice],
        reference_slot_ids=args.reference_slot_ids,
    )

    ref_features = pd.concat([ref_train, ref_cal, ref_test], axis=0).reset_index(drop=True)

    slot_prior, slot_prior_df, global_delta = build_slot_delta_prior(
        train_slots=slots[train_slice],
        train_run_value=y_run[train_slice],
        train_ref_features=ref_train,
        reference_slot_ids=args.reference_slot_ids,
        min_count=args.slot_delta_min_count,
        fallback=args.slot_delta_fallback,
    )

    log_subsection("Slot-delta prior learned from TRAIN only")
    print(f"slot_delta_min_count={args.slot_delta_min_count}")
    print(f"global_delta_run_median={global_delta:.6f}")
    print_df(slot_prior_df, max_rows=80)

    slot_delta_all = slot_delta_for_slots(slots, slot_prior, global_delta)
    slot_delta_iqr_all = slot_prior_iqr_for_slots(slots, slot_prior_df)

    ref_features["slot_delta_prior_run"] = slot_delta_all.astype(np.float32)
    ref_features["slot_delta_iqr"] = slot_delta_iqr_all.astype(np.float32)

    # Rule predictions for cal/test.
    slot_rule_run_cal, slot_rule_loop_cal = slot_delta_rule_loop(
        ref_run_median=ref_cal["ref_run_median"].to_numpy(dtype=np.float32),
        slot_delta=slot_delta_all[cal_slice],
    )
    slot_rule_run_test, slot_rule_loop_test = slot_delta_rule_loop(
        ref_run_median=ref_test["ref_run_median"].to_numpy(dtype=np.float32),
        slot_delta=slot_delta_all[test_slice],
    )

    trend_rule_run_cal, trend_rule_loop_cal = trend_rule_loop(
        trend_pred_run=ref_cal["ref_trend_pred_run"].to_numpy(dtype=np.float32),
    )
    trend_rule_run_test, trend_rule_loop_test = trend_rule_loop(
        trend_pred_run=ref_test["ref_trend_pred_run"].to_numpy(dtype=np.float32),
    )

    nonref_cal = ~np.isin(slots[cal_slice], args.reference_slot_ids)
    nonref_test = ~np.isin(slots[test_slice], args.reference_slot_ids)

    # Feature matrix base.
    drop_cols = {args.target_col, args.time_col, args.lot_col}

    if args.wafer_id_col in df.columns:
        drop_cols.add(args.wafer_id_col)

    if not args.include_slot_as_feature:
        drop_cols.add(args.slot_col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_base = df[feature_cols].copy()

    if args.include_slot_as_feature and args.slot_col not in X_base.columns:
        X_base["__slot_id_numeric"] = pd.to_numeric(df[args.slot_col], errors="coerce").astype(np.float32)

    X_base = coerce_mixed_columns_for_tabpfn(X_base)

    # Trend-enhanced feature matrix for F route.
    X_trend = X_base.copy()

    for c in ref_features.columns:
        X_trend[f"__trend_{c}"] = ref_features[c].astype(np.float32)

    train_mask_all = np.zeros(n_total, dtype=bool)
    train_mask_all[:train_end] = True

    ref_med_all = ref_features["ref_run_median"].to_numpy(dtype=np.float32)
    y_delta_run = y_run - ref_med_all
    train_mask_delta = train_mask_all & np.isfinite(y_delta_run)

    route_specs = [
        {
            "model_name": "C_loop_count",
            "target_type": "loop",
            "X": X_base,
            "y_target": y_loop.astype(np.float32),
            "train_mask": train_mask_all,
        },
        {
            "model_name": "F_delta_run_trend",
            "target_type": "delta_run_trend",
            "X": X_trend,
            "y_target": y_delta_run,
            "train_mask": train_mask_delta,
        },
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    safe = dataset_name.replace("/", "_").replace(" ", "_").replace(".", "_")

    full_rows = []
    selected_rows = []

    log_section("Training selected routes and selecting high-conf rules on CAL")

    for spec in route_specs:
        model_name = spec["model_name"]
        target_type = spec["target_type"]

        log_subsection(f"Fit/Predict {model_name}")
        print(f"  principle: {ROUTE_INFO[model_name]['principle']}")
        print(f"  implementation_diff: {ROUTE_INFO[model_name]['implementation_diff']}")

        pred_cal_raw, pred_test_raw, selected_cols, fs_info = fit_predict_tabpfn(
            model_name=model_name,
            X_all=spec["X"],
            y_target=spec["y_target"],
            train_mask=spec["train_mask"],
            cal_slice=cal_slice,
            test_slice=test_slice,
            args=args,
        )

        detail_cal = build_prediction_detail(
            model_name=model_name,
            target_type=target_type,
            pred_raw=pred_cal_raw,
            y_true_met=y_met[cal_slice],
            y_true_run=y_run[cal_slice],
            y_true_loop=y_loop[cal_slice],
            lots=lots[cal_slice],
            slots=slots[cal_slice],
            ref_features=ref_features.iloc[cal_slice].reset_index(drop=True),
            slot_rule_run=slot_rule_run_cal,
            slot_rule_loop_arr=slot_rule_loop_cal,
            trend_rule_run=trend_rule_run_cal,
            trend_rule_loop_arr=trend_rule_loop_cal,
            nonref_mask=nonref_cal,
            reference_slot_ids=args.reference_slot_ids,
            use_residual_compensation=args.use_residual_compensation,
        )

        detail_test = build_prediction_detail(
            model_name=model_name,
            target_type=target_type,
            pred_raw=pred_test_raw,
            y_true_met=y_met[test_slice],
            y_true_run=y_run[test_slice],
            y_true_loop=y_loop[test_slice],
            lots=lots[test_slice],
            slots=slots[test_slice],
            ref_features=ref_features.iloc[test_slice].reset_index(drop=True),
            slot_rule_run=slot_rule_run_test,
            slot_rule_loop_arr=slot_rule_loop_test,
            trend_rule_run=trend_rule_run_test,
            trend_rule_loop_arr=trend_rule_loop_test,
            nonref_mask=nonref_test,
            reference_slot_ids=args.reference_slot_ids,
            use_residual_compensation=args.use_residual_compensation,
        )

        full_cal = evaluate_detail_full(detail_cal, args.diff_penalty_power)
        full_test = evaluate_detail_full(detail_test, args.diff_penalty_power)

        print_metric_block(f"CAL FULL {model_name}", full_cal)
        print_metric_block(f"TEST FULL {model_name}", full_test)

        full_rows.append(
            {
                "model": model_name,
                "target_type": target_type,
                "principle": ROUTE_INFO[model_name]["principle"],
                "implementation_diff": ROUTE_INFO[model_name]["implementation_diff"],
                "cal_full_n": full_cal["n"] if full_cal else 0,
                "cal_full_acc": full_cal["accuracy"] if full_cal else np.nan,
                "cal_full_severe": full_cal["severe_ge2"] if full_cal else np.nan,
                "test_full_n": full_test["n"] if full_test else 0,
                "test_full_acc": full_test["accuracy"] if full_test else np.nan,
                "test_full_within1": full_test["within_1"] if full_test else np.nan,
                "test_full_severe": full_test["severe_ge2"] if full_test else np.nan,
                "test_full_penalty": full_test["weighted_penalty"] if full_test else np.nan,
                "n_features_used": len(selected_cols),
            }
        )

        # CAL scan -> select -> TEST apply.
        for mode in ["slot", "trend", "both", "either"]:
            scan_cal = high_conf_scan_on_detail(
                detail=detail_cal,
                model_name=model_name,
                target_type=target_type,
                mode=mode,
                penalty_power=args.diff_penalty_power,
                min_scan_n=args.min_cal_hc_n,
                hc_min_ref=args.hc_min_ref,
                hc_max_comp_shift=args.hc_max_comp_shift,
            )

            scan_path = os.path.join(args.output_dir, f"{safe}_{model_name}_{mode}_cal_scan.csv")
            scan_cal.to_csv(scan_path, index=False)

            selected_rule, select_reason = select_rule_from_cal_scan(
                scan_cal,
                target_acc=args.target_hc_acc,
                max_severe=args.max_hc_severe,
            )

            if selected_rule is None:
                selected_rows.append(
                    {
                        "model": model_name,
                        "target_type": target_type,
                        "mode": mode,
                        "select_reason": select_reason,
                        "principle": ROUTE_INFO[model_name]["principle"],
                        "implementation_diff": ROUTE_INFO[model_name]["implementation_diff"],
                    }
                )
                continue

            selected_rule = dict(selected_rule)
            selected_rule["model"] = model_name
            selected_rule["target_type"] = target_type
            selected_rule["mode"] = mode
            selected_rule["select_reason"] = select_reason
            selected_rule["hc_min_ref"] = args.hc_min_ref
            selected_rule["hc_max_comp_shift"] = args.hc_max_comp_shift

            cal_metrics, cal_mask = evaluate_detail_rule(
                detail_cal,
                selected_rule,
                args.diff_penalty_power,
            )
            test_metrics, test_mask = evaluate_detail_rule(
                detail_test,
                selected_rule,
                args.diff_penalty_power,
            )

            cal_total = int((detail_cal["is_nonref"] & detail_cal["valid"]).sum())
            test_total = int((detail_test["is_nonref"] & detail_test["valid"]).sum())

            cal_cov = 100.0 * int(cal_mask.sum()) / max(cal_total, 1)
            test_cov = 100.0 * int(test_mask.sum()) / max(test_total, 1)

            selected_rows.append(
                {
                    **selected_rule,
                    "principle": ROUTE_INFO[model_name]["principle"],
                    "implementation_diff": ROUTE_INFO[model_name]["implementation_diff"],
                    "cal_selected_n": int(cal_mask.sum()),
                    "cal_selected_coverage": cal_cov,
                    "cal_selected_acc": cal_metrics["accuracy"] if cal_metrics else np.nan,
                    "cal_selected_within1": cal_metrics["within_1"] if cal_metrics else np.nan,
                    "cal_selected_severe": cal_metrics["severe_ge2"] if cal_metrics else np.nan,
                    "cal_selected_penalty": cal_metrics["weighted_penalty"] if cal_metrics else np.nan,
                    "test_applied_n": int(test_mask.sum()),
                    "test_applied_coverage": test_cov,
                    "test_applied_acc": test_metrics["accuracy"] if test_metrics else np.nan,
                    "test_applied_within1": test_metrics["within_1"] if test_metrics else np.nan,
                    "test_applied_severe": test_metrics["severe_ge2"] if test_metrics else np.nan,
                    "test_applied_penalty": test_metrics["weighted_penalty"] if test_metrics else np.nan,
                    "test_warning_small_n": int(test_mask.sum()) < args.min_test_hc_n_warn,
                }
            )

            test_detail_selected = detail_test.copy()
            test_detail_selected["selected_by_cal_rule"] = test_mask
            test_detail_selected.to_csv(
                os.path.join(args.output_dir, f"{safe}_{model_name}_{mode}_test_detail_selected.csv"),
                index=False,
            )

        detail_cal.to_csv(os.path.join(args.output_dir, f"{safe}_{model_name}_cal_detail.csv"), index=False)
        detail_test.to_csv(os.path.join(args.output_dir, f"{safe}_{model_name}_test_detail.csv"), index=False)

    full_df = pd.DataFrame(full_rows)
    selected_df = pd.DataFrame(selected_rows)

    full_path = os.path.join(args.output_dir, f"{safe}_summary_full_cal_test.csv")
    selected_path = os.path.join(args.output_dir, f"{safe}_summary_cal_selected_test_applied.csv")
    slot_prior_path = os.path.join(args.output_dir, f"{safe}_slot_delta_prior.csv")

    full_df.to_csv(full_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    slot_prior_df.to_csv(slot_prior_path, index=False)

    log_section("SUMMARY 1: Full CAL/TEST for selected routes")
    print_df(full_df.sort_values("test_full_acc", ascending=False), max_rows=20)

    log_section("SUMMARY 2: CAL-selected high-conf rules applied to TEST")
    if len(selected_df):
        cols_order = [
            "model",
            "target_type",
            "mode",
            "select_reason",
            "principle",
            "implementation_diff",
            "cal_selected_n",
            "cal_selected_coverage",
            "cal_selected_acc",
            "cal_selected_severe",
            "test_applied_n",
            "test_applied_coverage",
            "test_applied_acc",
            "test_applied_within1",
            "test_applied_severe",
            "test_applied_penalty",
            "test_warning_small_n",
            "boundary_min",
            "abs_ref_bias_loop_max",
            "rule_run_gap_max",
            "ref_trend_abs_corr_min",
            "ref_run_std_max",
            "slot_delta_iqr_max",
        ]
        cols = [c for c in cols_order if c in selected_df.columns]
        print_df(
            selected_df[cols].sort_values(
                ["test_applied_acc", "test_applied_coverage"],
                ascending=[False, False],
            ),
            max_rows=80,
        )
    else:
        print("  <empty>")

    print("\nSaved CSV:")
    print(f"  full_summary={full_path}")
    print(f"  selected_rules_summary={selected_path}")
    print(f"  slot_prior={slot_prior_path}")

    return {
        "dataset": dataset_name,
        "full_df": full_df,
        "selected_df": selected_df,
    }


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-route calibrated high-confidence evaluation: C_loop_count and F_delta_run_trend."
    )

    p.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    p.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    p.add_argument("--time-col", type=str, default=DEFAULT_TIME_COL)
    p.add_argument("--slot-col", type=str, default=DEFAULT_SLOT_COL)
    p.add_argument("--lot-col", type=str, default=DEFAULT_LOT_COL)
    p.add_argument("--wafer-id-col", type=str, default=DEFAULT_WAFER_ID_COL)

    p.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    p.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)

    p.add_argument("--reference-slot-ids", type=str, default=DEFAULT_REFERENCE_SLOT_IDS)

    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    p.add_argument("--softmax-temperature", type=float, default=DEFAULT_SOFTMAX_TEMPERATURE)
    p.add_argument("--average-before-softmax", action="store_true", default=DEFAULT_AVERAGE_BEFORE_SOFTMAX)
    p.add_argument("--poly-features", type=int, default=DEFAULT_POLY_FEATURES)
    p.add_argument("--subsample-samples", type=int, default=DEFAULT_SUBSAMPLE_SAMPLES)
    p.add_argument("--predict-batch-size", type=int, default=DEFAULT_PREDICT_BATCH_SIZE)

    p.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    p.add_argument("--max-missing-ratio", type=float, default=DEFAULT_MAX_MISSING_RATIO)
    p.add_argument("--min-variance", type=float, default=DEFAULT_MIN_VARIANCE)

    p.add_argument("--label-out-of-range", type=str, choices=["clip", "nan", "error"], default="clip")
    p.add_argument("--diff-penalty-power", type=float, default=DEFAULT_DIFF_PENALTY_POWER)

    p.add_argument("--target-hc-acc", type=float, default=DEFAULT_TARGET_HC_ACC)
    p.add_argument("--max-hc-severe", type=float, default=DEFAULT_MAX_HC_SEVERE)
    p.add_argument("--min-cal-hc-n", type=int, default=DEFAULT_MIN_CAL_HC_N)
    p.add_argument("--min-test-hc-n-warn", type=int, default=DEFAULT_MIN_TEST_HC_N_WARN)

    p.add_argument("--hc-min-ref", type=int, default=2)
    p.add_argument("--hc-max-comp-shift", type=int, default=1)

    p.add_argument("--slot-delta-min-count", type=int, default=DEFAULT_SLOT_DELTA_MIN_COUNT)
    p.add_argument("--slot-delta-fallback", type=float, default=DEFAULT_SLOT_DELTA_FALLBACK)

    p.add_argument("--include-slot-as-feature", action="store_true", default=True)
    p.add_argument("--no-include-slot-as-feature", action="store_false", dest="include_slot_as_feature")

    p.add_argument("--use-residual-compensation", action="store_true", default=True)
    p.add_argument("--no-residual-compensation", action="store_false", dest="use_residual_compensation")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.reference_slot_ids = parse_reference_slot_ids(args.reference_slot_ids)

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, but model uses device='cuda'.")

    print(f"OUTPUT_DIR={args.output_dir}")
    print(f"DATA_PATH={args.data_path}")
    print(f"split train_ratio={args.train_ratio}, val_ratio={args.val_ratio}")
    print(f"reference_slot_ids={args.reference_slot_ids}")
    print(f"use_residual_compensation={args.use_residual_compensation}")
    print(f"include_slot_as_feature={args.include_slot_as_feature}")
    print(f"target_hc_acc={args.target_hc_acc}, max_hc_severe={args.max_hc_severe}, min_cal_hc_n={args.min_cal_hc_n}")
    print("Selected routes: C_loop_count, F_delta_run_trend")
    print("Rule selection: scan on CAL, apply fixed selected rule to TEST")

    files = discover_files(args.data_path)
    print(f"Found {len(files)} file(s).")

    all_results = []
    t_all = time.time()

    for i, fp in enumerate(files):
        print(f"\n[{i + 1}/{len(files)}] {fp}")
        df = load_single_file(fp)
        print(f"loaded shape={df.shape}")

        try:
            res = infer_one_dataset(df, os.path.basename(fp), args)
            if res is not None:
                all_results.append(res)
        except Exception as e:
            print(f"❌ failed: {type(e).__name__}: {e}")
        finally:
            del df
            force_cleanup(light=True)

    log_section("ALL DATASETS DONE")
    print(f"success={len(all_results)}/{len(files)} total_time={time.time() - t_all:.1f}s")

    if all_results:
        full_all = pd.concat([r["full_df"].assign(dataset=r["dataset"]) for r in all_results], axis=0)
        selected_all = pd.concat([r["selected_df"].assign(dataset=r["dataset"]) for r in all_results], axis=0)

        print("\nCombined full summary:")
        print_df(full_all, max_rows=120)

        print("\nCombined selected-rule test summary:")
        print_df(selected_all, max_rows=200)

        full_all.to_csv(os.path.join(args.output_dir, "ALL_summary_full_cal_test.csv"), index=False)
        selected_all.to_csv(os.path.join(args.output_dir, "ALL_summary_cal_selected_test_applied.csv"), index=False)

    print("\nProcess finished, exiting now...", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
