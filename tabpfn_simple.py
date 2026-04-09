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


def plot_pred_true_timeseries(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_is_ref: np.ndarray,
    title: str,
    out_path: str,
    ylabel: str,
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

    drop_cols = {target_col, time_col, slot_col, lot_col}
    if wafer_id_col in df.columns:
        drop_cols.add(wafer_id_col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    if not feature_cols:
        print(f"  ⚠️ skip {dataset_name}: no feature columns after dropping meta cols")
        return None

    t_prep0 = time.time()
    X_raw = df[feature_cols]
    y = df[target_col].astype(float).to_numpy(dtype=np.float32)
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
    model.fit(X_selected.iloc[:val_end], y[:val_end])
    t_fit = time.time() - t_fit0

    t_pred0 = time.time()
    y_pred_raw = predict_maybe_batched(model, X_selected.iloc[val_end:], batch_size=predict_batch_size)
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
    y_test = y[val_end:]
    y_pred = apply_residual_compensation(
        df_meta=meta_test,
        y_true=y_test,
        y_pred=y_pred_raw,
        lot_col=lot_col,
        slot_col=slot_col,
        reference_slot_ids=reference_slot_ids,
    )
    t_comp = time.time() - t_comp0

    metrics = eval_metrics(y_test[test_is_nonref], y_pred[test_is_nonref])

    t_plot0 = time.time()
    safe = dataset_name.replace("/", "_").replace(" ", "_").replace(".", "_")
    plot_path = os.path.join(output_dir, f"{safe}_infer_timeseries.png")
    plot_pred_true_timeseries(
        y_test=y_test,
        y_pred=y_pred,
        test_is_ref=test_is_ref,
        title=(
            f"{dataset_name} | COMP Non-ref MAE={metrics['mae']:.4f} R²={metrics['r2']:.4f} "
            f"Acc@0.5={metrics['acc05']:.1f}% Acc@1.0={metrics['acc10']:.1f}%"
        ),
        out_path=plot_path,
        ylabel=target_col,
    )
    t_plot = time.time() - t_plot0

    print(
        f"  timing: sort={t_sort:.2f}s prep={t_prep:.2f}s "
        f"fit={t_fit:.2f}s pred={t_pred:.2f}s comp={t_comp:.2f}s plot={t_plot:.2f}s"
    )

    return {
        "dataset": dataset_name,
        "n_rows": int(n_total),
        "n_features_raw": int(len(feature_cols)),
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

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    reference_slot_ids = _parse_reference_slot_ids(args.reference_slot_ids)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    files = discover_files(args.data_path)
    print(f"Found {len(files)} file(s). OUTPUT_DIR={args.output_dir}")

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
            )
            if res is not None:
                all_results.append(res)
                m = res["metrics"]
                print(
                    f"  COMP Non-ref: MAE={m['mae']:.4f} R²={m['r2']:.4f} "
                    f"Acc@0.5={m['acc05']:.1f}% Acc@1.0={m['acc10']:.1f}% "
                    f"| time={res['time_sec']:.1f}s "
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
        print(
            f"AVG COMP Non-ref MAE={avg_mae:.4f} | "
            f"AVG Acc@0.5={avg_acc05:.1f}% | AVG Acc@1.0={avg_acc10:.1f}%"
        )


if __name__ == "__main__":
    main()
    print("Process finished, exiting now...", flush=True)
    os._exit(0)
