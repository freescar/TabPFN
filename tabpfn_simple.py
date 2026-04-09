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

from tabpfn import TabPFNRegressor
import posthog
posthog.disabled = True

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Degrees of freedom")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")


# ============================================================
# Defaults (keep aligned with previous hard-coded values)
# ============================================================

DEFAULT_DATA_PATH = "/ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229/EPLBAB01_CHA1_1011_1229.parquet"
DEFAULT_OUTPUT_DIR = "./results/EPLBAB01_CHA1_1101_1120_simple"

DEFAULT_TARGET_COL = "met"
DEFAULT_TIME_COL = "start_time"
DEFAULT_SLOT_COL = "slot_id"
DEFAULT_LOT_COL = "lot_id"
DEFAULT_WAFER_ID_COL = "wafer_id"

DEFAULT_REFERENCE_SLOT_IDS = "2,3,4,5,12,13,20,21,22,23"

DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.8

DEFAULT_MODEL_PATH = "/ossfs/workspace/xrfm/TabPFN-main/models/tabpfn-v2.5-regressor-v2.5_default.ckpt"

DEFAULT_N_ESTIMATORS = 32
DEFAULT_SOFTMAX_TEMPERATURE = 0.5
DEFAULT_AVERAGE_BEFORE_SOFTMAX = False

DEFAULT_POLY_FEATURES = 50
DEFAULT_SUBSAMPLE_SAMPLES = 10_000
DEFAULT_PREDICT_BATCH_SIZE = 200


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

def force_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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

    # ±0.5 band (visual for Acc@0.5)
    plt.fill_between(
        x,
        y_test - 0.5,
        y_test + 0.5,
        alpha=0.10,
        color="green",
        label="±0.5 band",
    )

    # True
    plt.plot(x, y_test, color="black", alpha=0.35, linewidth=1.0, label="true (all)")
    plt.scatter(x[is_nonref], y_test[is_nonref], s=8, color="black", alpha=0.6, label="true (non-ref)")
    plt.scatter(x[test_is_ref], y_test[test_is_ref], s=8, color="gray", alpha=0.4, label="true (ref)")

    # Pred (comp)
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
    """
    Residual compensation baseline:
      For each lot, compute bias = mean(y_true_ref - y_pred_ref) using reference wafers,
      then add the bias to non-reference wafers within the same lot.
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


def batched_predict(model: TabPFNRegressor, X: pd.DataFrame, batch_size: int) -> np.ndarray:
    """
    IMPORTANT: Keep X as a DataFrame to support string/categorical features.
    """
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
) -> dict | None:
    # required cols
    required = [target_col, slot_col, time_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ⚠️ skip {dataset_name}: missing {missing}")
        return None

    # sort by time
    df = df.sort_values(time_col, ascending=True).reset_index(drop=True)

    # lot_id for compensation
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

    # split
    _train_end = int(n_total * train_ratio)
    val_end = int(n_total * val_ratio)

    # feature columns = everything except meta/label/time
    drop_cols = {target_col, time_col, slot_col, lot_col}
    if wafer_id_col in df.columns:
        drop_cols.add(wafer_id_col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    if not feature_cols:
        print(f"  ⚠️ skip {dataset_name}: no feature columns after dropping meta cols")
        return None

    # X as DataFrame (keep strings/categorical)
    X = df[feature_cols].copy()

    # numeric inf -> NaN only
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)

    # y numeric
    y = df[target_col].astype(float).to_numpy(dtype=np.float32)

    slots = df[slot_col].to_numpy()
    test_is_ref = np.isin(slots[val_end:], reference_slot_ids)
    test_is_nonref = ~test_is_ref
    if test_is_nonref.sum() == 0:
        print(f"  ⚠️ skip {dataset_name}: no non-ref rows in test")
        return None

    # fit(train+val) -> predict(test)
    t0 = time.time()
    model = create_model(
        model_path=model_path,
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        average_before_softmax=average_before_softmax,
        poly_features=poly_features,
        subsample_samples=subsample_samples,
    )
    model.fit(X.iloc[:val_end], y[:val_end])
    y_pred_raw = batched_predict(model, X.iloc[val_end:], batch_size=predict_batch_size)
    del model
    force_cleanup()
    infer_time = time.time() - t0

    # compensation
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

    metrics = eval_metrics(y_test[test_is_nonref], y_pred[test_is_nonref])

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

    return {
        "dataset": dataset_name,
        "n_rows": int(n_total),
        "n_features": int(len(feature_cols)),
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
        description="TabPFN baseline inference with per-lot residual compensation using reference slots."
    )

    # data / output
    p.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH, help="Input file or folder (csv/parquet).")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output folder for plots/results.")

    # column names
    p.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    p.add_argument("--time-col", type=str, default=DEFAULT_TIME_COL)
    p.add_argument("--slot-col", type=str, default=DEFAULT_SLOT_COL)
    p.add_argument("--lot-col", type=str, default=DEFAULT_LOT_COL)
    p.add_argument("--wafer-id-col", type=str, default=DEFAULT_WAFER_ID_COL)

    # split
    p.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    p.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)

    # reference slots
    p.add_argument(
        "--reference-slot-ids",
        type=str,
        default=DEFAULT_REFERENCE_SLOT_IDS,
        help='Comma-separated slot ids, e.g. "2,3,4,5,12,13,20,21,22,23".',
    )

    # model / inference config
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
    p.add_argument("--predict-batch-size", type=int, default=DEFAULT_PREDICT_BATCH_SIZE)

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
            )
            if res is not None:
                all_results.append(res)
                m = res["metrics"]
                print(
                    f"  COMP Non-ref: MAE={m['mae']:.4f} R²={m['r2']:.4f} "
                    f"Acc@0.5={m['acc05']:.1f}% Acc@1.0={m['acc10']:.1f}% "
                    f"| time={res['time_sec']:.1f}s"
                )
                print(f"  plot={res['plot']}")
        except Exception as e:
            print(f"  ❌ failed: {e}")
        finally:
            del df
            force_cleanup()

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
