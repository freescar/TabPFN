#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["TABPFN_NO_TELEMETRY"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["DISABLE_POSTHOG"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["SEGMENT_WRITE_KEY"] = ""
os.environ["ANALYTICS_DISABLED"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import gc
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from tabpfn import TabPFNClassifier


# ============================================================
# 业务公式 (参考 tabpfn_simple_probalistic_loop.py)
# ============================================================
FB_DC_TARGET1 = 81.0
PRE_OFFSET = 0.3127
REC1_GRADIENT = 0.1313
LOOP_OFFSET = 6.0

RUN_VALUE_BOUNDS = np.array([0.0, 19.5, 26.2, 33.0, 39.8, 46.5, 53.5, 60.1, 100.0], dtype=np.float64)
CLASS_LABELS = np.arange(2, 10, dtype=int)


def ocd_to_run_value(ocd):
    """将 OCD 值转换为 run_value"""
    return (FB_DC_TARGET1 - np.asarray(ocd, dtype=np.float64) - PRE_OFFSET) / REC1_GRADIENT - LOOP_OFFSET


def run_value_to_loop(rv, out_of_range="clip"):
    """将 run_value 转换为 loop_count"""
    rv = np.asarray(rv, dtype=np.float64)
    idx = np.searchsorted(RUN_VALUE_BOUNDS[1:-1], rv, side="right")
    loop = CLASS_LABELS[np.clip(idx, 0, len(CLASS_LABELS) - 1)].astype(float)
    if out_of_range == "nan":
        loop[~((rv >= RUN_VALUE_BOUNDS[0]) & (rv < RUN_VALUE_BOUNDS[-1]))] = np.nan
    return loop


def ocd_to_loop(ocd, out_of_range="clip"):
    """从 OCD (GroundTruth) 计算 loop_count"""
    return run_value_to_loop(ocd_to_run_value(ocd), out_of_range=out_of_range)


def force_cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _parse_int_list(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_slot_ref_features(
    df: pd.DataFrame,
    *,
    target_col: str,
    slot_col: str,
    lot_col: str,
    reference_slot_ids: Sequence[int],
) -> pd.DataFrame:
    slots = df[slot_col].to_numpy(dtype=np.int32)
    lots = df[lot_col].to_numpy()
    target_vals = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32)
    n_rows = len(df)

    is_ref = np.isin(slots, reference_slot_ids)
    slot_min = float(slots.min())
    slot_max = float(slots.max())
    slot_range = max(slot_max - slot_min, 1.0)
    slot_norm = (slots - slot_min) / slot_range

    feat: dict[str, np.ndarray] = {
        "slot_id": slots.astype(np.float32),
        "slot_norm": slot_norm.astype(np.float32),
        "slot_center_dist": np.abs(slot_norm - 0.5).astype(np.float32),
        "slot_trend_sq": (slot_norm**2).astype(np.float32),
        "slot_trend_cubic": (slot_norm**3).astype(np.float32),
        "is_ref_slot": is_ref.astype(np.float32),
    }

    if len(reference_slot_ids) > 0:
        ref_ids_arr = np.array(sorted(reference_slot_ids), dtype=float)
        feat["nearest_ref_dist"] = np.min(np.abs(slots[:, None].astype(float) - ref_ids_arr[None, :]), axis=1).astype(
            np.float32
        )
    else:
        feat["nearest_ref_dist"] = np.zeros(n_rows, dtype=np.float32)

    lot_ref_mean = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_std = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_count = np.zeros(n_rows, dtype=np.float32)

    for lot in np.unique(lots):
        lot_mask = lots == lot
        lot_ref_mask = lot_mask & is_ref
        n_ref = int(lot_ref_mask.sum())
        if n_ref == 0:
            continue

        lot_ref_y = target_vals[lot_ref_mask]
        total_sum = float(np.nansum(lot_ref_y))
        total_sum2 = float(np.nansum(lot_ref_y**2))

        nonref_idx = np.where(lot_mask & (~is_ref))[0]
        if len(nonref_idx) > 0:
            mean_val = total_sum / n_ref
            var_val = max(total_sum2 / n_ref - mean_val**2, 0.0)
            lot_ref_mean[nonref_idx] = mean_val
            lot_ref_std[nonref_idx] = float(np.sqrt(var_val))
            lot_ref_count[nonref_idx] = float(n_ref)

        ref_idx = np.where(lot_ref_mask)[0]
        for idx in ref_idx:
            own = target_vals[idx]
            n_loo = n_ref - 1
            if n_loo <= 0:
                continue
            loo_mean = (total_sum - own) / n_loo
            loo_sum2 = total_sum2 - own**2
            loo_std = float(np.sqrt(max(loo_sum2 / n_loo - loo_mean**2, 0.0)))
            lot_ref_mean[idx] = loo_mean
            lot_ref_std[idx] = loo_std
            lot_ref_count[idx] = float(n_loo)

    feat["lot_ref_target_mean"] = lot_ref_mean
    feat["lot_ref_target_std"] = lot_ref_std
    feat["lot_ref_target_count"] = lot_ref_count

    X = pd.DataFrame(feat, index=df.index)
    for c in X.columns:
        if X[c].isna().any():
            med = X[c].median()
            if np.isnan(med):
                med = 0.0
            X[c] = X[c].fillna(med)
    return X.astype(np.float32)


def build_numeric_features(df: pd.DataFrame, *, exclude_cols: set[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in exclude_cols]
    if not cols:
        raise ValueError("No usable columns left for numeric features.")

    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    if X.shape[1] == 0:
        raise ValueError("All candidate feature columns became NaN after numeric conversion.")

    for c in X.columns:
        if X[c].isna().any():
            med = X[c].median()
            if not np.isfinite(med):
                med = 0.0
            X[c] = X[c].fillna(float(med))

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    return X.astype(np.float32)


def eval_class_control_metrics(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    *,
    class_labels: np.ndarray,
    penalty_power: float,
) -> dict:
    y_true_cls = np.asarray(y_true_cls, dtype=int)
    y_pred_cls = np.asarray(y_pred_cls, dtype=int)
    diff = y_pred_cls - y_true_cls
    abs_diff = np.abs(diff)
    max_diff = int(class_labels.max() - class_labels.min())

    weighted_penalty = float(np.mean(abs_diff.astype(np.float32) ** penalty_power))
    worst_penalty = float(max_diff**penalty_power) if max_diff > 0 else 1.0
    control_score = float(max(0.0, 100.0 * (1.0 - weighted_penalty / worst_penalty)))

    return {
        "accuracy": float(accuracy_score(y_true_cls, y_pred_cls) * 100.0),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_cls, y_pred_cls) * 100.0),
        "macro_f1": float(
            f1_score(y_true_cls, y_pred_cls, labels=class_labels, average="macro", zero_division=0) * 100.0
        ),
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


def eval_subset_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subset_mask: np.ndarray,
    *,
    class_labels: np.ndarray,
    penalty_power: float,
) -> dict:
    subset_mask = np.asarray(subset_mask, dtype=bool)
    n_samples = int(subset_mask.sum())
    if n_samples == 0:
        return {
            "n_samples": 0,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "within_1": float("nan"),
            "within_2": float("nan"),
            "mae_class": float("nan"),
            "rmse_class": float("nan"),
            "mean_signed_diff": float("nan"),
            "severe_diff_ge2": float("nan"),
            "extreme_diff_ge3": float("nan"),
            "weighted_penalty": float("nan"),
            "control_score": float("nan"),
        }
    base = eval_class_control_metrics(
        y_true[subset_mask],
        y_pred[subset_mask],
        class_labels=class_labels,
        penalty_power=penalty_power,
    )
    base["n_samples"] = n_samples
    return base


def predict_quantile_classes(
    proba: np.ndarray,
    classes: np.ndarray,
    q_lower: float,
    q_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(classes)
    sorted_classes = classes[order]
    sorted_proba = proba[:, order]
    cdf = np.cumsum(sorted_proba, axis=1)

    hit_lo = cdf >= q_lower
    hit_hi = cdf >= q_upper

    lo_idx = np.argmax(hit_lo, axis=1)
    hi_idx = np.argmax(hit_hi, axis=1)

    lo_no_hit = ~hit_lo.any(axis=1)
    hi_no_hit = ~hit_hi.any(axis=1)
    lo_idx[lo_no_hit] = 0
    hi_idx[hi_no_hit] = cdf.shape[1] - 1

    q_lo = sorted_classes[lo_idx]
    q_hi = sorted_classes[hi_idx]
    return q_lo.astype(np.float32), q_hi.astype(np.float32)


def confidence_score_from_proba(
    proba: np.ndarray,
    classes: np.ndarray,
    ci_width: np.ndarray,
) -> np.ndarray:
    classes = classes.astype(np.float32)
    expected_class = (proba * classes[None, :]).sum(axis=1)

    boundaries = np.arange(classes.min() + 0.5, classes.max(), 1.0, dtype=np.float32)
    if len(boundaries) > 0:
        margin = np.min(np.abs(expected_class[:, None] - boundaries[None, :]), axis=1)
    else:
        margin = np.ones_like(expected_class, dtype=np.float32)

    half_width = np.maximum(ci_width * 0.5, 1e-6)
    score = margin / half_width
    return np.where(np.isfinite(score), score, -np.inf).astype(np.float32)


def make_classifier(
    model_path: str,
    n_estimators: int,
    softmax_temperature: float,
    poly_features: int,
    subsample_samples: int,
) -> TabPFNClassifier:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabPFNClassifier(
        model_path=model_path,
        device=device,
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        average_before_softmax=True,
        memory_saving_mode=True,
        ignore_pretraining_limits=True,
        inference_config={
            "SUBSAMPLE_SAMPLES": max(256, int(subsample_samples)),
            "POLYNOMIAL_FEATURES": max(1, int(poly_features)),
        },
    )


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    ref_slots = _parse_int_list(args.reference_slot_ids)
    conf_width_thresholds = [float(x.strip()) for x in args.conf_width_thresholds.split(",") if x.strip()]
    coverage_levels = [float(x.strip()) for x in args.coverage_levels.split(",") if x.strip()]

    if args.ci_quantile_lower >= args.ci_quantile_upper:
        raise ValueError("ci-quantile-lower must be smaller than ci-quantile-upper")

    df = load_data(args.data_path)
    if args.time_col in df.columns:
        df = df.sort_values(args.time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    for required in [args.target_col, args.slot_col]:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")

    # 如果 target_col 是 'GroundTruth'，则计算 loop_count
    use_ground_truth = (args.target_col == 'GroundTruth')
    if use_ground_truth:
        print(f"[INFO] Detected target_col='GroundTruth', computing loop_count from OCD values...")
        ground_truth_vals = pd.to_numeric(df[args.target_col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(ground_truth_vals)):
            raise ValueError(f"Target column {args.target_col} contains NaN or inf")
        # 计算 loop_count
        df['loop_count'] = ocd_to_loop(ground_truth_vals, out_of_range="clip").astype(int)
        print(f"[INFO] Computed loop_count from GroundTruth. Distribution:")
        print(df['loop_count'].value_counts().sort_index())
        # 使用计算出的 loop_count 作为目标
        actual_target_col = 'loop_count'
    else:
        actual_target_col = args.target_col

    if args.use_reference_slot_features and args.lot_col not in df.columns:
        if args.wafer_id_col in df.columns:
            df[args.lot_col] = df[args.wafer_id_col].astype(str).str[:-2]
        else:
            raise ValueError("lot-col missing and wafer-id-col not available for fallback lot extraction")

    n = len(df)
    split = int(n * args.val_ratio)
    if split <= 0 or split >= n:
        raise ValueError(f"Invalid split by val-ratio={args.val_ratio}, n={n}")

    y_raw = pd.to_numeric(df[actual_target_col], errors="coerce").to_numpy(dtype=np.float32)
    valid_target = np.isfinite(y_raw)
    if not np.all(valid_target):
        raise ValueError(f"Target column {actual_target_col} contains NaN or inf")

    y = np.rint(y_raw).astype(int)

    # 构建排除列集合，如果使用 GroundTruth，则排除它
    exclude_cols = {
        actual_target_col,
        args.time_col,
        args.slot_col,
        args.lot_col,
        args.wafer_id_col,
    }
    if use_ground_truth:
        exclude_cols.add('GroundTruth')
        print(f"[INFO] Excluding 'GroundTruth' from features during training/inference")

    if args.use_reference_slot_features:
        # 使用实际的目标列（计算出的 loop_count 或原始 target_col）进行参考片特征工程
        # 但如果是 GroundTruth，我们需要用 GroundTruth 的值来计算参考片统计
        ref_target_col = args.target_col if use_ground_truth else actual_target_col
        X = build_slot_ref_features(
            df,
            target_col=ref_target_col,
            slot_col=args.slot_col,
            lot_col=args.lot_col,
            reference_slot_ids=ref_slots,
        )
        if args.include_raw_numeric_features:
            X_num = build_numeric_features(df, exclude_cols=exclude_cols)
            X = pd.concat([X, X_num], axis=1)
            X = X.loc[:, ~X.columns.duplicated()].copy()
    else:
        X = build_numeric_features(df, exclude_cols=exclude_cols)

    X_train = X.iloc[:split]
    y_train = y[:split]
    X_test = X.iloc[split:]
    y_test = y[split:]

    model = make_classifier(
        model_path=args.model_path,
        n_estimators=args.n_estimators,
        softmax_temperature=args.softmax_temperature,
        poly_features=args.poly_features,
        subsample_samples=args.subsample_samples,
    )

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    classes = np.asarray(model.classes_)

    y_pred = classes[np.argmax(proba, axis=1)].astype(int)
    q_lower, q_upper = predict_quantile_classes(proba, classes, args.ci_quantile_lower, args.ci_quantile_upper)
    ci_width = q_upper - q_lower

    test_slots = df[args.slot_col].to_numpy()[split:]
    test_is_ref = np.isin(test_slots, ref_slots) if ref_slots else np.zeros(len(X_test), dtype=bool)

    eval_mask = ~test_is_ref if ref_slots else np.ones_like(test_is_ref, dtype=bool)
    if eval_mask.sum() == 0:
        raise ValueError("No non-reference rows in test split for evaluation")

    class_labels = np.unique(y_train)
    full_metrics = eval_subset_metrics(
        y_true=y_test,
        y_pred=y_pred,
        subset_mask=eval_mask,
        class_labels=class_labels,
        penalty_power=args.diff_penalty_power,
    )

    ci_threshold_metrics: dict[str, dict] = {}
    for thr in conf_width_thresholds:
        key = f"ci_thr{thr:.1f}"
        subset = eval_mask & (ci_width <= thr)
        m = eval_subset_metrics(
            y_true=y_test,
            y_pred=y_pred,
            subset_mask=subset,
            class_labels=class_labels,
            penalty_power=args.diff_penalty_power,
        )
        m["threshold"] = float(thr)
        m["coverage_pct"] = float(np.mean(subset) * 100.0)
        ci_threshold_metrics[key] = m

    score = confidence_score_from_proba(proba, classes.astype(np.float32), ci_width.astype(np.float32))
    nonref_idx = np.flatnonzero(eval_mask)
    sorted_nonref_idx = nonref_idx[np.argsort(-score[nonref_idx], kind="stable")]

    top_coverage_metrics: dict[str, dict] = {}
    for cov in coverage_levels:
        cov_pct = int(round(cov * 100))
        key = f"cov{cov_pct}"

        subset_size = int(np.ceil(cov * len(nonref_idx))) if len(nonref_idx) > 0 else 0
        subset_idx = sorted_nonref_idx[:subset_size]
        subset_mask = np.zeros_like(eval_mask, dtype=bool)
        subset_mask[subset_idx] = True

        m = eval_subset_metrics(
            y_true=y_test,
            y_pred=y_pred,
            subset_mask=subset_mask,
            class_labels=class_labels,
            penalty_power=args.diff_penalty_power,
        )
        m["target_coverage_pct"] = float(cov * 100.0)
        m["achieved_coverage_pct"] = float(np.mean(subset_mask) * 100.0)
        m["subset_size"] = int(subset_size)
        m["confidence_score_mean"] = float(np.mean(score[subset_mask])) if subset_size > 0 else float("nan")
        m["ci_width_mean"] = float(np.mean(ci_width[subset_mask])) if subset_size > 0 else float("nan")
        top_coverage_metrics[key] = m

    pred_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "q_lower": q_lower,
            "q_upper": q_upper,
            "ci_width": ci_width,
            "is_ref": test_is_ref,
            "confidence_score": score,
        }
    )

    for i, c in enumerate(classes):
        pred_df[f"proba_{int(c)}"] = proba[:, i]

    if args.lot_col in df.columns:
        pred_df[args.lot_col] = df[args.lot_col].to_numpy()[split:]
    if args.slot_col in df.columns:
        pred_df[args.slot_col] = test_slots
    
    # 如果使用了 GroundTruth，也保存原始 OCD 值供参考
    if use_ground_truth:
        pred_df['ground_truth_ocd'] = df[args.target_col].to_numpy()[split:]

    pred_path = os.path.join(args.output_dir, "loop_count_test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    summary = {
        "n_total": int(n),
        "n_train": int(split),
        "n_test": int(n - split),
        "n_features": int(X.shape[1]),
        "use_reference_slot_features": bool(args.use_reference_slot_features),
        "computed_from_ground_truth": bool(use_ground_truth),
        **{f"full_{k}": v for k, v in full_metrics.items()},
    }
    summary_path = os.path.join(args.output_dir, "loop_count_summary_metrics.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    ci_rows = []
    for key, m in ci_threshold_metrics.items():
        ci_rows.append({"bucket": key, **m})
    pd.DataFrame(ci_rows).to_csv(os.path.join(args.output_dir, "loop_count_ci_threshold_metrics.csv"), index=False)

    cov_rows = []
    for key, m in top_coverage_metrics.items():
        cov_rows.append({"bucket": key, **m})
    pd.DataFrame(cov_rows).to_csv(os.path.join(args.output_dir, "loop_count_top_coverage_metrics.csv"), index=False)

    print(f"[INFO] rows={n} train={split} test={n-split}")
    print(f"[INFO] features={X.shape[1]} use_reference_slot_features={args.use_reference_slot_features}")
    if use_ground_truth:
        print(f"[INFO] loop_count computed from GroundTruth (OCD values)")
    print(
        f"[FULL] Acc={full_metrics['accuracy']:.2f}% BalancedAcc={full_metrics['balanced_accuracy']:.2f}% "
        f"MacroF1={full_metrics['macro_f1']:.2f}% Within1={full_metrics['within_1']:.2f}% "
        f"Severe(|d|>=2)={full_metrics['severe_diff_ge2']:.2f}% Score={full_metrics['control_score']:.2f}"
    )

    for thr in conf_width_thresholds:
        key = f"ci_thr{thr:.1f}"
        m = ci_threshold_metrics[key]
        print(
            f"  CI<= {thr:.1f}: coverage={m['coverage_pct']:.2f}% n={m['n_samples']} "
            f"Acc={m['accuracy']:.2f}% Within1={m['within_1']:.2f}% "
            f"Severe(|d|>=2)={m['severe_diff_ge2']:.2f}% Score={m['control_score']:.2f}"
        )

    for cov in coverage_levels:
        cov_pct = int(round(cov * 100))
        key = f"cov{cov_pct}"
        m = top_coverage_metrics[key]
        print(
            f"  Top confidence {cov_pct}%: achieved={m['achieved_coverage_pct']:.2f}% n={m['subset_size']} "
            f"Acc={m['accuracy']:.2f}% Within1={m['within_1']:.2f}% "
            f"Severe(|d|>=2)={m['severe_diff_ge2']:.2f}% Score={m['control_score']:.2f}"
        )

    print(f"[OUT] {summary_path}")
    print(f"[OUT] {pred_path}")

    del model
    force_cleanup()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Baseline probabilistic loop_count classification with TabPFNClassifier (multiclass checkpoint)."
    )
    p.add_argument("--data-path", type=str, default="/ossfs/workspace/tools/A2_DBJOA_BW09_Tool06_CHA.csv")
    p.add_argument("--output-dir", type=str, default="./results/baseline_loop_count_classifier")

    p.add_argument("--target-col", type=str, default="loop_count",
                   help="Target column name. Use 'GroundTruth' to compute loop_count from OCD values, or 'loop_count' to use directly.")
    p.add_argument("--time-col", type=str, default="start_time")
    p.add_argument("--slot-col", type=str, default="slot_id")
    p.add_argument("--lot-col", type=str, default="lot_id")
    p.add_argument("--wafer-id-col", type=str, default="wafer_id")

    p.add_argument("--val-ratio", type=float, default=0.8)

    p.add_argument(
        "--model-path",
        type=str,
        default="./models/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt",
    )
    p.add_argument("--n-estimators", type=int, default=4)
    p.add_argument("--softmax-temperature", type=float, default=0.9)
    p.add_argument("--poly-features", type=int, default=1)
    p.add_argument("--subsample-samples", type=int, default=2048)

    p.add_argument("--ci-quantile-lower", type=float, default=0.1)
    p.add_argument("--ci-quantile-upper", type=float, default=0.9)
    p.add_argument("--conf-width-thresholds", type=str, default="1.0,2.0,3.0")
    p.add_argument("--coverage-levels", type=str, default="0.1,0.2,0.3")
    p.add_argument("--diff-penalty-power", type=float, default=2.0)

    p.add_argument("--reference-slot-ids", type=str, default="2,3,4,5,12,13,20,21,22,23")
    p.add_argument("--disable-reference-slot-features", dest="use_reference_slot_features", action="store_false")
    p.add_argument("--include-raw-numeric-features", action="store_true", default=False)

    p.set_defaults(use_reference_slot_features=True)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
