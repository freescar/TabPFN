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
from sklearn.metrics import mean_absolute_error, r2_score

from tabpfn import TabPFNRegressor


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
    conf_width_thresholds: Sequence[float],
) -> dict:
    base = eval_metrics(y_true, y_pred)

    ci_width = q_upper - q_lower
    empirical_coverage = float(np.mean((y_true >= q_lower) & (y_true <= q_upper)) * 100.0)

    base.update(
        {
            "ci_width_mean": float(np.mean(ci_width)),
            "ci_width_median": float(np.median(ci_width)),
            "ci_empirical_coverage_pct": empirical_coverage,
        }
    )

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
    mets = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32)
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

    lot_ref_met_mean = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_std = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_median = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_min = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_max = np.full(n_rows, np.nan, dtype=np.float32)
    lot_ref_met_count = np.zeros(n_rows, dtype=np.float32)
    ref_met_interp = np.full(n_rows, np.nan, dtype=np.float32)

    ref_slot_mets: dict[int, np.ndarray] = {
        sid: np.full(n_rows, np.nan, dtype=np.float32) for sid in reference_slot_ids
    }

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
        total_sum2 = float(np.nansum(lot_ref_mets**2))

        sort_order = np.argsort(lot_ref_slots)
        sorted_ref_slots = lot_ref_slots[sort_order].astype(float)
        sorted_ref_mets = lot_ref_mets[sort_order]

        nonref_indices = np.where(lot_mask & ~is_ref)[0]
        if len(nonref_indices) > 0:
            mean_val = total_sum / n_ref
            var_val = max(total_sum2 / n_ref - mean_val**2, 0.0)
            lot_ref_met_mean[nonref_indices] = mean_val
            lot_ref_met_std[nonref_indices] = float(np.sqrt(var_val))
            lot_ref_met_median[nonref_indices] = float(np.nanmedian(lot_ref_mets))
            lot_ref_met_min[nonref_indices] = float(np.nanmin(lot_ref_mets))
            lot_ref_met_max[nonref_indices] = float(np.nanmax(lot_ref_mets))
            lot_ref_met_count[nonref_indices] = float(n_ref)
            ref_met_interp[nonref_indices] = np.interp(slots[nonref_indices], sorted_ref_slots, sorted_ref_mets).astype(
                np.float32
            )

        ref_indices = np.where(lot_ref_mask)[0]
        for idx in ref_indices:
            curr_slot = int(slots[idx])
            own_met = slot_met_dict.get(curr_slot)
            if own_met is None:
                continue
            n_loo = n_ref - 1
            if n_loo <= 0:
                continue
            loo_sum = total_sum - own_met
            loo_mean = loo_sum / n_loo
            loo_sum2 = total_sum2 - own_met**2
            loo_std = float(np.sqrt(max(loo_sum2 / n_loo - loo_mean**2, 0.0)))
            loo_mets = np.array([m for s, m in slot_met_dict.items() if s != curr_slot], dtype=np.float32)

            lot_ref_met_mean[idx] = loo_mean
            lot_ref_met_std[idx] = loo_std
            lot_ref_met_median[idx] = float(np.nanmedian(loo_mets))
            lot_ref_met_min[idx] = float(np.nanmin(loo_mets))
            lot_ref_met_max[idx] = float(np.nanmax(loo_mets))
            lot_ref_met_count[idx] = float(n_loo)

            loo_items = [(s, slot_met_dict[s]) for s in slot_met_dict if s != curr_slot]
            loo_slots = np.array([s for s, _ in loo_items], dtype=float)
            loo_vals = np.array([m for _, m in loo_items], dtype=np.float32)
            if len(loo_slots) > 0:
                order = np.argsort(loo_slots)
                ref_met_interp[idx] = float(np.interp(float(curr_slot), loo_slots[order], loo_vals[order]))

            ref_slot_mets[curr_slot][idx] = np.nan

    feat.update(
        {
            "lot_ref_met_mean": lot_ref_met_mean,
            "lot_ref_met_std": lot_ref_met_std,
            "lot_ref_met_median": lot_ref_met_median,
            "lot_ref_met_min": lot_ref_met_min,
            "lot_ref_met_max": lot_ref_met_max,
            "lot_ref_met_range": lot_ref_met_max - lot_ref_met_min,
            "lot_ref_met_count": lot_ref_met_count,
            "ref_met_interp": ref_met_interp,
        }
    )

    for sid in reference_slot_ids:
        feat[f"ref_slot_{sid}_met"] = ref_slot_mets[sid]
        feat[f"ref_slot_{sid}_met_dev"] = ref_slot_mets[sid] - lot_ref_met_mean

    result = pd.DataFrame(feat, index=df.index)
    for col in result.columns:
        if result[col].isna().any():
            med = result[col].median()
            if np.isnan(med):
                med = 0.0
            result[col] = result[col].fillna(med)

    return result.astype(np.float32)


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


def residual_compensation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lots: np.ndarray,
    slots: np.ndarray,
    reference_slot_ids: Sequence[int],
) -> np.ndarray:
    compensated = y_pred.copy()
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


def make_regressor(
    model_path: str,
    n_estimators: int,
    softmax_temperature: float,
    poly_features: int,
    subsample_samples: int,
) -> TabPFNRegressor:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabPFNRegressor(
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

    if args.use_reference_slot_features and args.lot_col not in df.columns:
        if args.wafer_id_col in df.columns:
            df[args.lot_col] = df[args.wafer_id_col].astype(str).str[:-2]
        else:
            raise ValueError("lot-col missing and wafer-id-col not available for fallback lot extraction")

    n = len(df)
    split = int(n * args.val_ratio)
    if split <= 0 or split >= n:
        raise ValueError(f"Invalid split by val-ratio={args.val_ratio}, n={n}")

    y = pd.to_numeric(df[args.target_col], errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(y).any():
        raise ValueError(f"Target column {args.target_col} contains NaN after numeric conversion")

    exclude_cols = {
        args.target_col,
        args.time_col,
        args.slot_col,
        args.lot_col,
        args.wafer_id_col,
    }

    if args.use_reference_slot_features:
        X = build_slot_ref_features(
            df,
            target_col=args.target_col,
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

    model = make_regressor(
        model_path=args.model_path,
        n_estimators=args.n_estimators,
        softmax_temperature=args.softmax_temperature,
        poly_features=args.poly_features,
        subsample_samples=args.subsample_samples,
    )

    X_train = X.iloc[:split]
    y_train = y[:split]
    X_test = X.iloc[split:]
    y_test = y[split:]

    model.fit(X_train, y_train)

    pred = model.predict(
        X_test,
        output_type="main",
        quantiles=[args.ci_quantile_lower, args.ci_quantile_upper],
    )
    y_pred_raw = np.asarray(pred["mean"], dtype=np.float32)
    q_lower_raw = np.asarray(pred["quantiles"][0], dtype=np.float32)
    q_upper_raw = np.asarray(pred["quantiles"][1], dtype=np.float32)

    test_slots = df[args.slot_col].to_numpy()[split:]
    test_is_ref = np.isin(test_slots, ref_slots) if ref_slots else np.zeros(len(X_test), dtype=bool)

    if args.use_reference_slot_compensation and ref_slots:
        lots_test = df[args.lot_col].to_numpy()[split:]
        y_pred = residual_compensation(y_test, y_pred_raw, lots_test, test_slots, ref_slots)
    else:
        y_pred = y_pred_raw.copy()

    bias_shift = y_pred - y_pred_raw
    q_lower = q_lower_raw + bias_shift
    q_upper = q_upper_raw + bias_shift

    eval_mask = ~test_is_ref if ref_slots else np.ones_like(test_is_ref, dtype=bool)
    if eval_mask.sum() == 0:
        raise ValueError("No non-reference rows in test split for evaluation")

    metrics = eval_metrics_prob(
        y_true=y_test[eval_mask],
        y_pred=y_pred[eval_mask],
        q_lower=q_lower[eval_mask],
        q_upper=q_upper[eval_mask],
        conf_width_thresholds=conf_width_thresholds,
    )

    results_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "q_lower": q_lower,
            "q_upper": q_upper,
            "ci_width": q_upper - q_lower,
            "is_ref": test_is_ref,
        }
    )

    if args.lot_col in df.columns:
        results_df[args.lot_col] = df[args.lot_col].to_numpy()[split:]
    if args.slot_col in df.columns:
        results_df[args.slot_col] = test_slots

    summary_path = os.path.join(args.output_dir, "regression_summary_metrics.csv")
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)

    pred_path = os.path.join(args.output_dir, "regression_test_predictions.csv")
    results_df.to_csv(pred_path, index=False)

    print(f"[INFO] rows={n} train={split} test={n-split}")
    print(f"[INFO] features={X.shape[1]} use_reference_slot_features={args.use_reference_slot_features}")
    print(f"[METRICS] MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f} Acc@0.5={metrics['acc05']:.2f}% Acc@1.0={metrics['acc10']:.2f}%")
    print(
        f"[METRICS] CI width mean={metrics['ci_width_mean']:.4f} median={metrics['ci_width_median']:.4f} "
        f"coverage={metrics['ci_empirical_coverage_pct']:.2f}%"
    )
    for thr in conf_width_thresholds:
        key = f"ci_thr{thr:.1f}"
        print(
            f"  CI<= {thr:.1f}: coverage={metrics[f'{key}_coverage_pct']:.2f}% "
            f"MAE={metrics[f'{key}_mae']:.4f} R2={metrics[f'{key}_r2']:.4f} "
            f"Acc@0.5={metrics[f'{key}_acc05']:.2f}% Acc@1.0={metrics[f'{key}_acc10']:.2f}%"
        )
    print(f"[OUT] {summary_path}")
    print(f"[OUT] {pred_path}")

    del model
    force_cleanup()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Baseline probabilistic Target regression with TabPFNRegressor (mediumdata checkpoint)."
    )
    p.add_argument("--data-path", type=str, default="/ossfs/workspace/tools/A2_DBJOA_BW09_Tool06_CHA.csv")
    p.add_argument("--output-dir", type=str, default="./results/baseline_regression")

    p.add_argument("--target-col", type=str, default="GroundTruth")
    p.add_argument("--time-col", type=str, default="start_time")
    p.add_argument("--slot-col", type=str, default="slot_id")
    p.add_argument("--lot-col", type=str, default="lot_id")
    p.add_argument("--wafer-id-col", type=str, default="wafer_id")

    p.add_argument("--val-ratio", type=float, default=0.8)

    p.add_argument(
        "--model-path",
        type=str,
        default="./models/tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt",
    )
    p.add_argument("--n-estimators", type=int, default=4)
    p.add_argument("--softmax-temperature", type=float, default=0.9)
    p.add_argument("--poly-features", type=int, default=1)
    p.add_argument("--subsample-samples", type=int, default=2048)

    p.add_argument("--ci-quantile-lower", type=float, default=0.1)
    p.add_argument("--ci-quantile-upper", type=float, default=0.9)
    p.add_argument("--conf-width-thresholds", type=str, default="1.0,2.0,3.0")

    p.add_argument("--reference-slot-ids", type=str, default="2,3,4,5,12,13,20,21,22,23")
    p.add_argument("--disable-reference-slot-features", dest="use_reference_slot_features", action="store_false")
    p.add_argument("--disable-reference-slot-compensation", dest="use_reference_slot_compensation", action="store_false")
    p.add_argument("--include-raw-numeric-features", action="store_true", default=False)

    p.set_defaults(use_reference_slot_features=True, use_reference_slot_compensation=True)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
