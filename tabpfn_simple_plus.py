import os
os.environ["TABPFN_NO_TELEMETRY"] = "1"

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
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
DEFAULT_OUTPUT_DIR = "./results/EPLBAB01_CHA1_1101_1120_simple_plus"

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

# final_y等级定义：diff=1影响小，diff>=2影响显著放大，默认用二次惩罚评估
CLASS_LABELS = np.arange(2, 10, dtype=int)
RUN_VALUE_BOUNDS = np.array([0.0, 19.5, 26.2, 33.0, 39.8, 46.5, 53.5, 60.1, 100.0], dtype=np.float32)
DEFAULT_DIFF_PENALTY_POWER = 2.0


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
# Label transform / Metrics / Plot
# ============================================================

def met_to_run_value(y: np.ndarray) -> np.ndarray:
    """Convert original met label to run_value by business formula."""
    y = np.asarray(y, dtype=np.float32)
    return (81.0 - y - 0.3127) / 0.1313 - 6.0


def run_value_to_final_y(run_value: np.ndarray, *, out_of_range: str = "clip") -> np.ndarray:
    """
    Convert run_value to control coefficient final_y in {2, ..., 9}.

    Intervals follow the requested left-closed/right-open rules:
      [0,19.5)->2, [19.5,26.2)->3, ..., [60.1,100)->9

    out_of_range:
      - clip: values <0 map to 2, values >=100 map to 9 (default, robust for production batches)
      - nan: out-of-range values become np.nan and can be filtered before training/eval
      - error: raise if any out-of-range value exists
    """
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
    """TabPFNRegressor outputs continuous values; convert to valid final_y class labels."""
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float32)
    return np.clip(np.rint(y_pred_cont), CLASS_LABELS[0], CLASS_LABELS[-1]).astype(int)


def eval_class_control_metrics(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    *,
    penalty_power: float,
) -> dict:
    """
    Classification metrics for ordered control coefficients.

    Key idea: final_y is ordinal. A miss by 1 is usually tolerable, while misses by 2+
    should be penalized much more. Therefore the main score uses |diff|**penalty_power.
    With penalty_power=2, diff=2 costs 4x diff=1, diff=3 costs 9x diff=1.
    """
    y_true_cls = np.asarray(y_true_cls, dtype=int)
    y_pred_cls = np.asarray(y_pred_cls, dtype=int)
    diff = y_pred_cls - y_true_cls
    abs_diff = np.abs(diff)
    max_diff = int(CLASS_LABELS[-1] - CLASS_LABELS[0])

    weighted_penalty = float(np.mean(abs_diff.astype(np.float32) ** penalty_power))
    worst_penalty = float(max_diff ** penalty_power)
    control_score = float(max(0.0, 100.0 * (1.0 - weighted_penalty / worst_penalty)))

    return {
        "accuracy": float(accuracy_score(y_true_cls, y_pred_cls) * 100.0),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_cls, y_pred_cls) * 100.0),
        "macro_f1": float(f1_score(y_true_cls, y_pred_cls, labels=CLASS_LABELS, average="macro", zero_division=0) * 100.0),
        "within_1": float(np.mean(abs_diff <= 1) * 100.0),
        "within_2": float(np.mean(abs_diff <= 2) * 100.0),
        "mae_class": float(mean_absolute_error(y_true_cls, y_pred_cls)),
        "rmse_class": float(np.sqrt(mean_squared_error(y_true_cls, y_pred_cls))),
        "mean_signed_diff": float(np.mean(diff)),
        "severe_diff_ge2": float(np.mean(abs_diff >= 2) * 100.0),
        "extreme_diff_ge3": float(np.mean(abs_diff >= 3) * 100.0),
        "weighted_penalty": weighted_penalty,
        "control_score": control_score,
    }


def per_class_metrics(y_true_cls: np.ndarray, y_pred_cls: np.ndarray) -> pd.DataFrame:
    rows = []
    y_true_cls = np.asarray(y_true_cls, dtype=int)
    y_pred_cls = np.asarray(y_pred_cls, dtype=int)
    for cls in CLASS_LABELS:
        mask = y_true_cls == cls
        n = int(mask.sum())
        if n == 0:
            rows.append({"class": int(cls), "support": 0, "accuracy": np.nan, "within_1": np.nan, "mae_class": np.nan})
            continue
        abs_diff = np.abs(y_pred_cls[mask] - y_true_cls[mask])
        rows.append(
            {
                "class": int(cls),
                "support": n,
                "accuracy": float(np.mean(abs_diff == 0) * 100.0),
                "within_1": float(np.mean(abs_diff <= 1) * 100.0),
                "mae_class": float(np.mean(abs_diff)),
            }
        )
    return pd.DataFrame(rows)


def plot_class_timeseries(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    test_is_ref: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    x = np.arange(len(y_true_cls))
    is_nonref = ~test_is_ref

    plt.figure(figsize=(18, 6))
    plt.plot(x, y_true_cls, color="black", alpha=0.45, linewidth=1.0, label="true final_y (all)")
    plt.scatter(x[is_nonref], y_true_cls[is_nonref], s=10, color="black", alpha=0.65, label="true non-ref")
    plt.scatter(x[test_is_ref], y_true_cls[test_is_ref], s=10, color="gray", alpha=0.35, label="true ref")

    plt.plot(x, y_pred_cls, color="steelblue", alpha=0.65, linewidth=1.2, label="pred final_y (comp)")
    plt.scatter(x[is_nonref], y_pred_cls[is_nonref], s=10, color="steelblue", alpha=0.7, label="pred non-ref")
    plt.scatter(x[test_is_ref], y_pred_cls[test_is_ref], s=10, color="salmon", alpha=0.35, label="pred ref")

    # ±1容忍带：用于展示小diff和大diff的差异
    plt.fill_between(x, y_true_cls - 1, y_true_cls + 1, alpha=0.10, color="green", label="±1 acceptable band")

    plt.yticks(CLASS_LABELS)
    plt.ylim(CLASS_LABELS[0] - 0.7, CLASS_LABELS[-1] + 0.7)
    plt.title(title)
    plt.xlabel("test sample index (time order)")
    plt.ylabel("final_y class / control coefficient")
    plt.grid(alpha=0.25)
    plt.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_confusion_matrix(y_true_cls: np.ndarray, y_pred_cls: np.ndarray, title: str, out_path: str) -> None:
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=CLASS_LABELS)
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized ratio")

    ax.set(
        xticks=np.arange(len(CLASS_LABELS)),
        yticks=np.arange(len(CLASS_LABELS)),
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        xlabel="predicted final_y",
        ylabel="true final_y",
        title=title,
    )

    thresh = cm_norm.max() / 2.0 if cm_norm.size else 0.5
    for i in range(len(CLASS_LABELS)):
        for j in range(len(CLASS_LABELS)):
            text = f"{cm[i, j]}\n{cm_norm[i, j] * 100:.0f}%"
            ax.text(j, i, text, ha="center", va="center", color="white" if cm_norm[i, j] > thresh else "black", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_diff_penalty(y_true_cls: np.ndarray, y_pred_cls: np.ndarray, title: str, out_path: str, penalty_power: float) -> None:
    diff = np.asarray(y_pred_cls, dtype=int) - np.asarray(y_true_cls, dtype=int)
    abs_diff = np.abs(diff)
    bins = np.arange(-7.5, 8.5, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(diff, bins=bins, color="steelblue", alpha=0.75, edgecolor="black")
    axes[0].axvline(0, color="black", linewidth=1.2)
    axes[0].axvspan(-1, 1, color="green", alpha=0.10, label="|diff|<=1")
    axes[0].set_xticks(np.arange(-7, 8, 1))
    axes[0].set_title("Signed diff distribution (pred - true)")
    axes[0].set_xlabel("diff")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    diff_levels = np.arange(0, 8, 1)
    counts = np.array([(abs_diff == d).sum() for d in diff_levels], dtype=int)
    penalties = diff_levels.astype(np.float32) ** penalty_power
    axes[1].bar(diff_levels, counts, color="salmon", alpha=0.75, edgecolor="black", label="count")
    ax2 = axes[1].twinx()
    ax2.plot(diff_levels, penalties, color="darkred", marker="o", linewidth=2.0, label=f"penalty=|diff|^{penalty_power:g}")
    axes[1].set_xticks(diff_levels)
    axes[1].set_title("Absolute diff count and penalty curve")
    axes[1].set_xlabel("|diff|")
    axes[1].set_ylabel("count")
    ax2.set_ylabel("single-sample penalty")
    axes[1].grid(alpha=0.25)

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def apply_residual_compensation(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lot_col: str,
    slot_col: str,
    reference_slot_ids: list[int],
) -> np.ndarray:
    """
    Per-lot residual compensation on continuous final_y prediction.

    The model predicts continuous ordinal final_y values first. We compensate reference-slot
    residuals in continuous space, then round/clip to final classes for evaluation/control.
    """
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
# Fast feature pruning
# ============================================================

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

    # 4) 单变量打分，选 top-K。final_y是有序类别，仍可作为连续序数做快速相关性筛选。
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
    label_out_of_range: str,
    diff_penalty_power: float,
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

    n_total_raw = len(df)
    if n_total_raw < 50:
        print(f"  ⚠️ skip {dataset_name}: too small n={n_total_raw}")
        return None

    y_met_raw = df[target_col].astype(float).to_numpy(dtype=np.float32)
    y_final_raw, run_value_raw, n_out_range = met_to_final_y(y_met_raw, out_of_range=label_out_of_range)
    if n_out_range > 0:
        print(f"  label transform: out-of-range run_value={n_out_range}/{n_total_raw} policy={label_out_of_range}")

    valid_mask = np.isfinite(y_final_raw)
    if valid_mask.sum() < n_total_raw:
        dropped = int(n_total_raw - valid_mask.sum())
        print(f"  label transform: drop {dropped} rows with invalid final_y")
        df = df.loc[valid_mask].reset_index(drop=True)
        lot_ids = lot_ids.loc[valid_mask].reset_index(drop=True)
        y_met_raw = y_met_raw[valid_mask]
        run_value_raw = run_value_raw[valid_mask]
        y_final_raw = y_final_raw[valid_mask]

    n_total = len(df)
    if n_total < 50:
        print(f"  ⚠️ skip {dataset_name}: too small after label filter n={n_total}")
        return None

    _train_end = int(n_total * train_ratio)
    val_end = int(n_total * val_ratio)

    drop_cols = {target_col, time_col, slot_col, lot_col}
    if wafer_id_col in df.columns:
        drop_cols.add(wafer_id_col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    if not feature_cols:
        print(f"  ⚠️ skip {dataset_name}: no feature columns after dropping meta cols")
        return None

    t_prep0 = time.time()
    X_raw = df[feature_cols]
    y_final = y_final_raw.astype(np.float32)
    X_raw = _coerce_mixed_columns_for_tabpfn(X_raw)

    X_train_for_select = X_raw.iloc[:val_end]
    X_selected, selected_cols, fs_info = fast_select_features(
        X_train=X_train_for_select,
        y_train=y_final[:val_end],
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
        f"  final_y distribution train+ref-window={dict(zip(*np.unique(y_final[:val_end].astype(int), return_counts=True)))}"
    )
    print(
        f"  feature pruning: raw={fs_info['raw_features']} -> "
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
    model.fit(X_selected.iloc[:val_end], y_final[:val_end])
    t_fit = time.time() - t_fit0

    t_pred0 = time.time()
    y_pred_cont_raw = predict_maybe_batched(model, X_selected.iloc[val_end:], batch_size=predict_batch_size)
    t_pred = time.time() - t_pred0

    infer_time = t_fit + t_pred

    del model
    force_cleanup(light=True)

    t_comp0 = time.time()
    meta_test = pd.DataFrame(
        {
            lot_col: lot_ids.iloc[val_end:].to_numpy(),
            slot_col: slots[val_end:],
        }
    )
    y_test_cont = y_final[val_end:]
    y_pred_cont = apply_residual_compensation(
        df_meta=meta_test,
        y_true=y_test_cont,
        y_pred=y_pred_cont_raw,
        lot_col=lot_col,
        slot_col=slot_col,
        reference_slot_ids=reference_slot_ids,
    )
    t_comp = time.time() - t_comp0

    y_test_cls = y_test_cont.astype(int)
    y_pred_cls = round_clip_final_y(y_pred_cont)

    metrics = eval_class_control_metrics(
        y_test_cls[test_is_nonref],
        y_pred_cls[test_is_nonref],
        penalty_power=diff_penalty_power,
    )
    per_cls = per_class_metrics(y_test_cls[test_is_nonref], y_pred_cls[test_is_nonref])

    t_plot0 = time.time()
    safe = dataset_name.replace("/", "_").replace(" ", "_").replace(".", "_")
    timeseries_path = os.path.join(output_dir, f"{safe}_plus_class_timeseries.png")
    cm_path = os.path.join(output_dir, f"{safe}_plus_confusion_matrix.png")
    diff_path = os.path.join(output_dir, f"{safe}_plus_diff_penalty.png")
    per_class_path = os.path.join(output_dir, f"{safe}_plus_per_class_metrics.csv")

    plot_class_timeseries(
        y_true_cls=y_test_cls,
        y_pred_cls=y_pred_cls,
        test_is_ref=test_is_ref,
        title=(
            f"{dataset_name} | COMP Non-ref Acc={metrics['accuracy']:.1f}% "
            f"Within1={metrics['within_1']:.1f}% Severe(|d|>=2)={metrics['severe_diff_ge2']:.1f}% "
            f"Score={metrics['control_score']:.1f}"
        ),
        out_path=timeseries_path,
    )
    plot_confusion_matrix(
        y_true_cls=y_test_cls[test_is_nonref],
        y_pred_cls=y_pred_cls[test_is_nonref],
        title=f"{dataset_name} | Non-ref confusion matrix",
        out_path=cm_path,
    )
    plot_diff_penalty(
        y_true_cls=y_test_cls[test_is_nonref],
        y_pred_cls=y_pred_cls[test_is_nonref],
        title=f"{dataset_name} | Non-ref diff penalty evaluation",
        out_path=diff_path,
        penalty_power=diff_penalty_power,
    )
    per_cls.to_csv(per_class_path, index=False)
    t_plot = time.time() - t_plot0

    print(
        f"  timing: sort={t_sort:.2f}s prep={t_prep:.2f}s "
        f"fit={t_fit:.2f}s pred={t_pred:.2f}s comp={t_comp:.2f}s plot={t_plot:.2f}s"
    )

    return {
        "dataset": dataset_name,
        "n_rows": int(n_total),
        "n_out_range": int(n_out_range),
        "n_features_raw": int(len(feature_cols)),
        "n_features_used": int(len(selected_cols)),
        "n_test": int(n_total - val_end),
        "n_test_nonref": int(test_is_nonref.sum()),
        "time_sec": float(infer_time),
        "metrics": metrics,
        "plots": {
            "timeseries": timeseries_path,
            "confusion_matrix": cm_path,
            "diff_penalty": diff_path,
            "per_class_csv": per_class_path,
        },
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
        description=(
            "TabPFN plus inference: transform met to ordinal final_y classes, then evaluate "
            "as control-oriented classification with stronger penalties for larger class diff."
        )
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

    p.add_argument(
        "--label-out-of-range",
        type=str,
        choices=["clip", "nan", "error"],
        default="clip",
        help="How to handle run_value outside [0,100): clip to edge class, drop as nan, or raise error.",
    )
    p.add_argument(
        "--diff-penalty-power",
        type=float,
        default=DEFAULT_DIFF_PENALTY_POWER,
        help="Penalty exponent for ordered class diff. 2 means quadratic penalty: diff=2 costs 4x diff=1.",
    )

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    reference_slot_ids = _parse_reference_slot_ids(args.reference_slot_ids)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    files = discover_files(args.data_path)
    print(f"Found {len(files)} file(s). OUTPUT_DIR={args.output_dir}")
    print(
        "Plus eval: met -> run_value -> final_y classes; metrics include exact accuracy, "
        "within_1, severe_diff_ge2, weighted_penalty, and control_score."
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
                label_out_of_range=args.label_out_of_range,
                diff_penalty_power=args.diff_penalty_power,
            )
            if res is not None:
                all_results.append(res)
                m = res["metrics"]
                print(
                    f"  COMP Non-ref: Acc={m['accuracy']:.1f}% BalancedAcc={m['balanced_accuracy']:.1f}% "
                    f"MacroF1={m['macro_f1']:.1f}% Within1={m['within_1']:.1f}% "
                    f"MAE_cls={m['mae_class']:.3f} RMSE_cls={m['rmse_class']:.3f} "
                    f"Severe(|d|>=2)={m['severe_diff_ge2']:.1f}% "
                    f"Penalty={m['weighted_penalty']:.3f} Score={m['control_score']:.1f} "
                    f"| time={res['time_sec']:.1f}s "
                    f"| features={res['n_features_raw']}->{res['n_features_used']}"
                )
                print(f"  plot.timeseries={res['plots']['timeseries']}")
                print(f"  plot.confusion={res['plots']['confusion_matrix']}")
                print(f"  plot.diff_penalty={res['plots']['diff_penalty']}")
                print(f"  csv.per_class={res['plots']['per_class_csv']}")
        except Exception as e:
            print(f"  ❌ failed: {type(e).__name__}: {e}")
        finally:
            del df
            force_cleanup(light=True)

    print(f"\nDone. success={len(all_results)}/{len(files)} total_time={time.time()-t_all:.1f}s")

    if all_results:
        avg = lambda key: float(np.mean([r["metrics"][key] for r in all_results]))
        print(
            f"AVG COMP Non-ref Acc={avg('accuracy'):.1f}% | "
            f"AVG Within1={avg('within_1'):.1f}% | "
            f"AVG Severe(|d|>=2)={avg('severe_diff_ge2'):.1f}% | "
            f"AVG Penalty={avg('weighted_penalty'):.3f} | "
            f"AVG ControlScore={avg('control_score'):.1f}"
        )


if __name__ == "__main__":
    main()
    print("Process finished, exiting now...", flush=True)
    os._exit(0)
