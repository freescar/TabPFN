import pandas as pd
import numpy as np
import gc
import torch
import warnings
import os
import time
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import differential_entropy

from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Degrees of freedom")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font.*")


# ============================================================
# 配置区
# ============================================================

data_path = '/ossfs/workspace/xrfm/TabPFN-main/datasets/WideTable-fdc_met_bw09_1011_1229'

TARGET_COL = 'met'
TIME_COL = "start_time"
train_end_ratio = 0.7
val_end_ratio = 0.8

MAX_FEATURES = 1000
MODEL_PATH = '/ossfs/workspace/xrfm/TabPFN-main/models/tabpfn-v2.5-regressor-v2.5_default.ckpt'

REFERENCE_SLOT_IDS = [2, 3, 4, 5, 12, 13, 20, 21, 22, 23]
SLOT_COL = "slot_id"
LOT_COL = "lot_id"
WAFER_ID_COL = "wafer_id"

OUTPUT_DIR = "/ossfs/workspace/xrfm/TabPFN-main/result/tool_all_fdc_1011_1229_mix"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 混合推理配置
TOOL_NAME_COL = "tool_name"          # 数据中区分 tool 的列名
RUN_MIXED_MODE = True                # True: 额外运行所有 tool 混合推理
MIXED_OUTPUT_DIR = "/ossfs/workspace/xrfm/TabPFN-main/result/tool_all_fdc_1011_1229_mix"
os.makedirs(MIXED_OUTPUT_DIR, exist_ok=True)

FIXED_CONFIG = {
    "n_estimators": 32,
    "softmax_temperature": 0.5,
    "average_before_softmax": False,
}
POLY_FEATURES = 20
PREDICT_BATCH_SIZE = 200

# rolling chunk
ROLLING_CHUNK_LOTS = 8

# =========================
# 稳定性 / 断点续跑配置
# =========================
ENABLE_RESUME = True
RESUME_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "single_tool_results.csv")

# Skip single-tool training if results exist
SKIP_SINGLE_TOOL_TRAINING = True  # Set to True to only run mixed inference

# 混合训练策略（v2）：
# 1. 按 lot 级别采样（保留 lot 内结构，确保 ref wafer 特征有效）
# 2. 每个 tool 贡献平衡（分层采样）
# 3. 更强的模型配置（n_estimators=16, poly=8）
# 4. tool_name 保留为类别特征（让 TabPFN 学习 tool-specific patterns）

# Mixed inference config (stronger for multi-tool complexity)
MIXED_MODEL_CONFIG = {
    "n_estimators": 16,       # ↑ from 8 (more ensemble for multi-tool noise)
    "polynomial_features": 8,  # ↑ from 1 (capture cross-tool interactions)
}

# Sampling strategy
MIXED_MAX_ROWS = 80_000                  # ↑ from 60_000 (more data for multi-tool)
MIXED_MAX_LOTS_PER_TOOL = 300            # ↓ from 450 (but keep FULL lots)
MIXED_MAX_NUMERIC_FEATURES = 600         # mixed 模式下数值特征上限（更保守）
MIXED_KEEP_TOOL_AS_CATEGORICAL = True    # Keep tool_name as categorical for TabPFN embeddings

# OOM 自动降配
OOM_ESTIMATORS_SCHEDULE = [32, 24, 16, 12, 8]
OOM_POLY_SCHEDULE = [20, 12, 8, 4, 1]

# rolling 自适应 chunk
ROLLING_CHUNK_LOTS_MIN = 1
ROLLING_CHUNK_LOTS_MAX = 8


# ============================================================
# 数据加载
# ============================================================

def load_single_file(filepath):
    if filepath.endswith('.parquet'):
        return pd.read_parquet(filepath)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {filepath}")


def discover_files(path):
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        parquet_files = sorted(glob.glob(os.path.join(path, '*.parquet')))
        csv_files = sorted(glob.glob(os.path.join(path, '*.csv')))
        all_files = parquet_files + csv_files
        if len(all_files) == 0:
            raise FileNotFoundError(f"文件夹 {path} 中没有 parquet/csv 文件")
        return all_files
    raise FileNotFoundError(f"路径不存在: {path}")


# ============================================================
# 工具函数
# ============================================================

def resolve_lot_column(df):
    if LOT_COL in df.columns:
        return LOT_COL
    if WAFER_ID_COL not in df.columns:
        raise ValueError(f"数据中既没有 '{LOT_COL}' 也没有 '{WAFER_ID_COL}' 列")
    df[LOT_COL] = df[WAFER_ID_COL].astype(str).str[:-2]
    return LOT_COL


def build_intra_lot_ref_features(X_df, y, meta_df, lot_col, slot_col,
                                  reference_slot_ids, selected_fdc_cols):
    lots = meta_df[lot_col].values
    slots = meta_df[slot_col].values
    is_ref = np.isin(slots, reference_slot_ids)
    n_rows = len(X_df)

    ref_y_mean = np.full(n_rows, np.nan)
    ref_y_std = np.full(n_rows, np.nan)
    ref_y_median = np.full(n_rows, np.nan)
    ref_y_min = np.full(n_rows, np.nan)
    ref_y_max = np.full(n_rows, np.nan)
    ref_y_range = np.full(n_rows, np.nan)
    ref_y_count = np.full(n_rows, 0.0)

    n_fdc = len(selected_fdc_cols)
    ref_fdc_means = np.full((n_rows, n_fdc), np.nan)
    fdc_col_indices = [list(X_df.columns).index(c) for c in selected_fdc_cols
                       if c in X_df.columns]
    X_np = X_df.values

    for lot in np.unique(lots):
        lot_mask = (lots == lot)
        lot_ref_mask = lot_mask & is_ref
        lot_indices = np.where(lot_mask)[0]
        lot_ref_indices = np.where(lot_ref_mask)[0]

        if lot_ref_mask.sum() == 0:
            continue

        ref_ys = y[lot_ref_indices]
        ref_fdcs = X_np[lot_ref_indices][:, fdc_col_indices] if fdc_col_indices else None

        for idx in lot_indices:
            if is_ref[idx]:
                other_ref = lot_ref_indices[lot_ref_indices != idx]
                if len(other_ref) == 0:
                    continue
                ys = y[other_ref]
                fdcs = X_np[other_ref][:, fdc_col_indices] if fdc_col_indices else None
            else:
                ys = ref_ys
                fdcs = ref_fdcs

            valid_ys = ys[~np.isnan(ys)]
            if len(valid_ys) == 0:
                continue

            ref_y_mean[idx] = np.mean(valid_ys)
            ref_y_std[idx] = np.std(valid_ys) if len(valid_ys) > 1 else 0.0
            ref_y_median[idx] = np.median(valid_ys)
            ref_y_min[idx] = np.min(valid_ys)
            ref_y_max[idx] = np.max(valid_ys)
            ref_y_range[idx] = np.max(valid_ys) - np.min(valid_ys)
            ref_y_count[idx] = len(valid_ys)

            if fdcs is not None and len(fdcs) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ref_fdc_means[idx] = np.nanmean(fdcs, axis=0)

    result = pd.DataFrame({
        "ref_y_mean": ref_y_mean, "ref_y_std": ref_y_std,
        "ref_y_median": ref_y_median, "ref_y_min": ref_y_min,
        "ref_y_max": ref_y_max, "ref_y_range": ref_y_range,
        "ref_y_count": ref_y_count,
    }, index=X_df.index)

    for i, col in enumerate(selected_fdc_cols):
        if i < ref_fdc_means.shape[1]:
            result[f"ref_fdc_mean_{col}"] = ref_fdc_means[:, i]

    for col in result.columns:
        col_mean = result[col].mean()
        if np.isnan(col_mean):
            col_mean = 0.0
        result[col] = result[col].fillna(col_mean)

    return result


def compute_entropy(series):
    vals = series.dropna()
    if len(vals) < 10:
        return -1e10
    if pd.api.types.is_numeric_dtype(series):
        arr = vals.values.astype(float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 10 or np.std(arr) == 0:
            return -1e10
        try:
            return float(differential_entropy(arr))
        except Exception:
            return -1e10
    else:
        probs = vals.value_counts(normalize=True).values
        return float(-np.sum(probs * np.log(probs + 1e-12)))


def staged_feature_selection_keep_categorical(
    X: pd.DataFrame,
    max_numeric_features: int = 1000,
    *,
    drop_categorical_all_nan: bool = True,
    drop_categorical_constant: bool = True,
    drop_categorical_high_cardinality: bool = True,
    high_cardinality_max_unique: int = 2000,
) -> pd.DataFrame:
    print(f"\n  特征筛选(保留类别) 总列={X.shape[1]} | 数值列目标 ≤ {max_numeric_features}")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    X_num = X[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X.index)
    X_cat = X[cat_cols].copy() if cat_cols else pd.DataFrame(index=X.index)

    if not X_num.empty:
        all_nan_cols = X_num.columns[X_num.isna().all()].tolist()
        if all_nan_cols:
            X_num = X_num.drop(columns=all_nan_cols)

        nunique = X_num.nunique(dropna=True)
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            X_num = X_num.drop(columns=const_cols)
            print(f"    [num] 删除 {len(const_cols)} 常数列 → {X_num.shape[1]}")

        nan_ratio = X_num.isna().mean()
        mostly_nan_cols = nan_ratio[nan_ratio > 0.9].index.tolist()
        if mostly_nan_cols:
            X_num = X_num.drop(columns=mostly_nan_cols)
            print(f"    [num] 删除 {len(mostly_nan_cols)} NaN>90%列 → {X_num.shape[1]}")

        if X_num.shape[1] > max_numeric_features:
            nan_counts = X_num.isna().sum()
            cols_with_nan = nan_counts[nan_counts > 0].sort_values(ascending=False)
            n_to_drop = X_num.shape[1] - max_numeric_features

            if n_to_drop <= len(cols_with_nan):
                X_num = X_num.drop(columns=cols_with_nan.index[:n_to_drop].tolist())
                print(f"    [num] 按NaN drop {n_to_drop} 列 → {X_num.shape[1]}")
            else:
                X_num = X_num.drop(columns=cols_with_nan.index.tolist())
                if X_num.shape[1] > max_numeric_features:
                    n_still = X_num.shape[1] - max_numeric_features
                    entropies = {col: compute_entropy(X_num[col]) for col in X_num.columns}
                    sorted_e = sorted(entropies.items(), key=lambda x: x[1])
                    X_num = X_num.drop(columns=[c for c, _ in sorted_e[:n_still]])
                    print(f"    [num] 按熵 drop {n_still} 列 → {X_num.shape[1]}")
    else:
        print("    [num] 数值列=0（跳过数值筛选）")

    if not X_cat.empty:
        before = X_cat.shape[1]

        if drop_categorical_all_nan:
            all_nan = X_cat.columns[X_cat.isna().all()].tolist()
            if all_nan:
                X_cat = X_cat.drop(columns=all_nan)

        if drop_categorical_constant and not X_cat.empty:
            nunique_cat = X_cat.nunique(dropna=True)
            const_cat = nunique_cat[nunique_cat <= 1].index.tolist()
            if const_cat:
                X_cat = X_cat.drop(columns=const_cat)

        if drop_categorical_high_cardinality and not X_cat.empty:
            nunique_cat = X_cat.nunique(dropna=True)
            high_card = nunique_cat[nunique_cat > high_cardinality_max_unique].index.tolist()
            if high_card:
                X_cat = X_cat.drop(columns=high_card)
                print(f"    [cat] 删除 {len(high_card)} 高基数列(nunique>{high_cardinality_max_unique})")

        after = X_cat.shape[1]
        print(f"    [cat] 保留 {after}/{before} 列")
    else:
        print("    [cat] 类别列=0")

    X_out = pd.concat([X_num, X_cat], axis=1)
    print(f"    ✅ 输出列: {X_out.shape[1]} (num={X_num.shape[1]} + cat={X_cat.shape[1]})")
    return X_out


def staged_feature_selection_mixed_mode(X: pd.DataFrame, max_numeric_features: int = 600) -> pd.DataFrame:
    """
    混合模式的特征筛选：
    - 保留 tool_name 作为类别特征（让 TabPFN 学习 tool embeddings）
    - 删除跨 tool 方差为 0 的特征（无区分度）
    - 优先保留高方差、低 NaN 的特征
    """
    print(f"\n  [Mixed] 特征筛选 总列={X.shape[1]} | 目标数值列 ≤ {max_numeric_features}")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    X_num = X[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X.index)
    X_cat = X[cat_cols].copy() if cat_cols else pd.DataFrame(index=X.index)

    if not X_num.empty:
        # Remove all-NaN columns
        all_nan_cols = X_num.columns[X_num.isna().all()].tolist()
        if all_nan_cols:
            X_num = X_num.drop(columns=all_nan_cols)

        # Remove constant columns
        nunique = X_num.nunique(dropna=True)
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            X_num = X_num.drop(columns=const_cols)
            print(f"    [num] 删除 {len(const_cols)} 常数列 → {X_num.shape[1]}")

        # Remove mostly-NaN (>95% for mixed mode, stricter than single-tool)
        nan_ratio = X_num.isna().mean()
        mostly_nan_cols = nan_ratio[nan_ratio > 0.95].index.tolist()
        if mostly_nan_cols:
            X_num = X_num.drop(columns=mostly_nan_cols)
            print(f"    [num] 删除 {len(mostly_nan_cols)} NaN>95%列 → {X_num.shape[1]}")

        # If still too many, prioritize features with high variance and low NaN
        if X_num.shape[1] > max_numeric_features:
            feature_scores = {}
            for col in X_num.columns:
                nan_count = X_num[col].isna().sum()
                variance = X_num[col].var()
                score = variance / (1 + nan_count / len(X_num))
                feature_scores[col] = score if not np.isnan(score) else -1e10

            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            keep_cols = [col for col, _ in sorted_features[:max_numeric_features]]
            X_num = X_num[keep_cols]
            print(f"    [num] 按方差/NaN评分保留 top {max_numeric_features} 列 → {X_num.shape[1]}")
    else:
        print("    [num] 数值列=0（跳过数值筛选）")

    if not X_cat.empty:
        # Always keep tool_name; filter other categoricals
        priority_cats = [c for c in X_cat.columns if c == TOOL_NAME_COL]
        other_cats = [c for c in X_cat.columns if c != TOOL_NAME_COL]

        filtered_cats = []
        for col in other_cats:
            col_nunique = X_cat[col].nunique(dropna=True)
            col_nan_ratio = X_cat[col].isna().mean()
            # Keep if: 2 ≤ unique ≤ 50 and NaN < 50%
            if 2 <= col_nunique <= 50 and col_nan_ratio < 0.5:
                filtered_cats.append(col)

        X_cat = X_cat[priority_cats + filtered_cats]
        print(f"    [cat] 保留 {len(X_cat.columns)} 列 (含 tool_name={TOOL_NAME_COL in X_cat.columns})")
    else:
        print("    [cat] 类别列=0")

    X_out = pd.concat([X_num, X_cat], axis=1)
    print(f"    ✅ 输出: {X_out.shape[1]} 列 (num={X_num.shape[1]} + cat={X_cat.shape[1]})")
    return X_out


def batched_predict(model, X, batch_size=None, **kwargs):
    if batch_size is None:
        batch_size = PREDICT_BATCH_SIZE
    results = []
    for i in range(0, len(X), batch_size):
        res = model.predict(X[i:i+batch_size], **kwargs)
        results.append(res)
    if isinstance(results[0], dict):
        merged = {}
        for key in results[0]:
            vals = [r[key] for r in results]
            if isinstance(vals[0], np.ndarray):
                merged[key] = np.concatenate(vals)
            elif isinstance(vals[0], list):
                n_q = len(vals[0])
                merged[key] = [np.concatenate([v[qi] for v in vals]) for qi in range(n_q)]
            else:
                merged[key] = vals[0]
        return merged
    elif isinstance(results[0], list):
        n_q = len(results[0])
        return [np.concatenate([r[qi] for r in results]) for qi in range(n_q)]
    else:
        return np.concatenate(results)


def create_model():
    return TabPFNRegressor(
        model_path=MODEL_PATH,
        device="cuda",
        n_estimators=FIXED_CONFIG["n_estimators"],
        softmax_temperature=FIXED_CONFIG["softmax_temperature"],
        average_before_softmax=FIXED_CONFIG["average_before_softmax"],
        memory_saving_mode=True,
        ignore_pretraining_limits=True,
        inference_config={
            "SUBSAMPLE_SAMPLES": 10_000,
            "POLYNOMIAL_FEATURES": POLY_FEATURES,
        },
    )


def create_model_with_params(n_estimators: int, poly_features: int):
    return TabPFNRegressor(
        model_path=MODEL_PATH,
        device="cuda",
        n_estimators=n_estimators,
        softmax_temperature=FIXED_CONFIG["softmax_temperature"],
        average_before_softmax=FIXED_CONFIG["average_before_softmax"],
        memory_saving_mode=True,
        ignore_pretraining_limits=True,
        inference_config={
            "SUBSAMPLE_SAMPLES": 10_000,
            "POLYNOMIAL_FEATURES": poly_features,
        },
    )


def is_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda oom" in msg) or ("cuda out of memory" in msg)


def fit_predict_with_oom_retry(X_train, y_train, X_test, stage_name="stage", mixed_mode=False):
    """
    OOM 时自动降 n_estimators / poly_features 重试。
    mixed_mode=True 时使用轻量配置 MIXED_MODEL_CONFIG 作为起点。
    """
    last_err = None
    if mixed_mode:
        ne0 = MIXED_MODEL_CONFIG["n_estimators"]
        poly0 = MIXED_MODEL_CONFIG["polynomial_features"]
        # Build a descending schedule starting from the mixed config values
        ne_sched = [ne for ne in OOM_ESTIMATORS_SCHEDULE if ne <= ne0] or [ne0]
        poly_sched = [poly0] * len(ne_sched)
        trials = list(zip(ne_sched, poly_sched))
    else:
        trials = list(zip(OOM_ESTIMATORS_SCHEDULE, OOM_POLY_SCHEDULE))
    for trial_i, (ne, poly) in enumerate(trials, 1):
        try:
            print(f"    [{stage_name}] trial {trial_i}/{len(trials)}: n_estimators={ne}, poly={poly}")
            model = create_model_with_params(ne, poly)
            model.fit(X_train, y_train)
            pred = batched_predict(model, X_test)
            del model
            force_cleanup()
            return pred, {"n_estimators": ne, "poly_features": poly}
        except Exception as e:
            last_err = e
            force_cleanup()
            if trial_i < len(trials):
                continue
            break
    raise RuntimeError(f"{stage_name} failed after retry schedule. last_err={last_err}")


def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_completed_datasets(summary_csv_path):
    if (not ENABLE_RESUME) or (not os.path.exists(summary_csv_path)):
        return set()
    try:
        hist = pd.read_csv(summary_csv_path)
        if "dataset" not in hist.columns:
            return set()
        return set(hist["dataset"].astype(str).tolist())
    except Exception:
        return set()


def append_result_to_summary(summary_csv_path, result_dict):
    row = {
        "dataset": result_dict["dataset"],
        "n_rows": result_dict["n_rows"],
        "n_lots": result_dict["n_lots"],
        "n_features": result_dict["n_features"],
        "n_test": result_dict["n_test"],
        "n_nonref": result_dict["n_nonref"],
        "mae": result_dict["results"]["Rolling raw"]["mae"],
        "r2": result_dict["results"]["Rolling raw"]["r2"],
        "acc05": result_dict["results"]["Rolling raw"]["acc05"],
        "acc10": result_dict["results"]["Rolling raw"]["acc10"],
        "baseline_time": result_dict["baseline_time"],
        "rolling_time": result_dict["rolling_time"],
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(summary_csv_path):
        df_row.to_csv(summary_csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(summary_csv_path, mode="w", header=True, index=False)


# ============================================================
# 指标 & 图（只画 raw）
# ============================================================

def compute_accuracy_at_threshold(y_true, y_pred, threshold):
    return float(np.mean(np.abs(y_true - y_pred) <= threshold) * 100)


def compute_all_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "acc05": compute_accuracy_at_threshold(y_true, y_pred, 0.5),
        "acc10": compute_accuracy_at_threshold(y_true, y_pred, 1.0),
    }


def plot_time_series_raw_only(y_test, baseline_raw, rolling_raw, test_is_ref,
                              results, dataset_name, output_dir, is_mixed_mode=False):
    safe_name = dataset_name.replace("/", "_").replace(" ", "_").replace(".", "_")
    n_test = len(y_test)
    x_axis = np.arange(n_test)
    test_is_nonref = ~test_is_ref

    if is_mixed_mode:
        # Mixed mode: only show baseline (rolling == baseline, no need for comparison)
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        axes_pairs = [(ax, ("Baseline raw", baseline_raw))]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        axes_pairs = list(zip(axes, [
            ("Baseline raw", baseline_raw),
            ("Rolling raw", rolling_raw),
        ]))

    for ax, (name, pred) in axes_pairs:
        r = results[name]

        ax.fill_between(x_axis, y_test - 0.5, y_test + 0.5,
                        alpha=0.12, color='green', label='±0.5 误差带')

        ax.plot(x_axis[test_is_nonref], y_test[test_is_nonref],
                'o', markersize=3, color='black', alpha=0.6, label='真实值 (non-ref)', zorder=3)
        ax.plot(x_axis[test_is_ref], y_test[test_is_ref],
                '^', markersize=3, color='gray', alpha=0.4, label='真实值 (ref)', zorder=3)

        ax.plot(x_axis[test_is_nonref], pred[test_is_nonref],
                'o', markersize=3, color='steelblue', alpha=0.6, label='预测值 (non-ref)', zorder=4)
        ax.plot(x_axis[test_is_ref], pred[test_is_ref],
                '^', markersize=3, color='salmon', alpha=0.4, label='预测值 (ref)', zorder=4)

        ax.plot(x_axis, y_test, '-', color='black', alpha=0.15, linewidth=0.5, zorder=1)
        ax.plot(x_axis, pred, '-', color='steelblue', alpha=0.15, linewidth=0.5, zorder=2)

        ax.set_ylabel('Met 值')
        ax.set_title(
            f'{name}  |  MAE={r["mae"]:.4f}  R²={r["r2"]:.4f}  '
            f'Acc@0.5={r["acc05"]:.1f}%  Acc@1.0={r["acc10"]:.1f}%',
            fontsize=11,
        )
        ax.legend(loc='upper right', fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

    if is_mixed_mode:
        ax.set_xlabel('样本序号（按时间顺序）')
    else:
        axes[-1].set_xlabel('样本序号（按时间顺序）')
    plt.suptitle(dataset_name, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'{safe_name}_timeseries_raw_only.png')
    plt.savefig(path, dpi=120)
    plt.close()
    return path


# ============================================================
# 核心 Pipeline（只跑 raw）
# ============================================================

def run_pipeline(df, dataset_name="dataset", output_dir=None, *, is_mixed_mode=False):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    print(f"\n{'─'*70}")
    print(f"  📂 {dataset_name}")
    print(f"{'─'*70}")

    df = df.sort_values(by=TIME_COL, ascending=True).reset_index(drop=True)
    lot_col = resolve_lot_column(df)

    for col in [SLOT_COL, TARGET_COL]:
        if col not in df.columns:
            print(f"  ❌ 缺少列 '{col}'，跳过")
            return None

    y_series = df[TARGET_COL].astype(float)
    if y_series.isna().all():
        print(f"  ❌ TARGET '{TARGET_COL}' 全为 NaN，跳过")
        return None

    is_ref_all = df[SLOT_COL].isin(REFERENCE_SLOT_IDS)
    n_ref_total = is_ref_all.sum()
    if n_ref_total == 0:
        print(f"  ❌ 没有 reference wafer (slots {REFERENCE_SLOT_IDS})，跳过")
        return None

    print(f"  数据: {len(df)} 行, {df[lot_col].nunique()} lots, "
          f"Ref: {n_ref_total} ({is_ref_all.mean()*100:.1f}%)")

    meta_cols = [lot_col, SLOT_COL]
    if WAFER_ID_COL in df.columns:
        meta_cols.append(WAFER_ID_COL)

    X_with_meta = df.drop(columns=[TARGET_COL])
    if TIME_COL in X_with_meta.columns:
        X_with_meta = X_with_meta.drop(columns=[TIME_COL])
    pure_feature_cols = [c for c in X_with_meta.columns if c not in meta_cols]

    if len(pure_feature_cols) < 5:
        print(f"  ❌ 特征数太少 ({len(pure_feature_cols)})，跳过")
        return None

    X_pure = X_with_meta[pure_feature_cols].copy()
    if is_mixed_mode:
        X_pure = staged_feature_selection_mixed_mode(X_pure, max_numeric_features=MIXED_MAX_NUMERIC_FEATURES)
    else:
        X_pure = staged_feature_selection_keep_categorical(X_pure, max_numeric_features=MAX_FEATURES)
    X_pure = X_pure.replace([np.inf, -np.inf], np.nan)
    selected_feature_cols = X_pure.columns.tolist()

    n_total = len(df)
    train_end = int(n_total * train_end_ratio)
    val_end = int(n_total * val_end_ratio)
    df_meta = df[[lot_col, SLOT_COL]].copy()

    n_test = n_total - val_end
    if n_test < 10:
        print(f"  ❌ 测试集太小 ({n_test} 条)，跳过")
        return None

    # 只从“数值列”里挑 top-k for ref fdc mean（避免字符串列导致 var 失败）
    top_k_fdc = 20
    train_mask = np.arange(n_total) < train_end
    numeric_fdc = [c for c in selected_feature_cols if pd.api.types.is_numeric_dtype(X_pure[c])]
    if len(numeric_fdc) > top_k_fdc:
        var = X_pure.loc[train_mask, numeric_fdc].var().sort_values(ascending=False)
        selected_fdc_for_ref = var.index[:top_k_fdc].tolist()
    else:
        selected_fdc_for_ref = numeric_fdc

    y_np_all = y_series.values.astype(np.float32)

    # ref 特征只用数值列构造（避免字符串）
    X_pure_num = X_pure[numeric_fdc].copy()
    ref_features = build_intra_lot_ref_features(
        X_pure_num, y_np_all, df_meta, lot_col, SLOT_COL,
        REFERENCE_SLOT_IDS, selected_fdc_for_ref,
    )

    X_merged = pd.concat([X_pure, ref_features], axis=1)
    print(f"  特征: {X_merged.shape[1]} 列 (base {X_pure.shape[1]} + ref {ref_features.shape[1]})")

    # 在 select_dtypes 之前，对类别列做 Label Encoding
    from sklearn.preprocessing import OrdinalEncoder
    cat_cols_in_merged = X_merged.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols_in_merged:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_merged[cat_cols_in_merged] = enc.fit_transform(X_merged[cat_cols_in_merged])

    # 仍然走纯数值输入
    X_all_np = X_merged.select_dtypes(include=[np.number]).values.astype(np.float32)
    X_all_np = np.nan_to_num(X_all_np, nan=np.nan, posinf=np.nan, neginf=np.nan)
    y_all_np = y_series.values.astype(np.float32)

    test_lots_ordered = df_meta.iloc[val_end:][lot_col].unique()
    lot_groups = df_meta.groupby(lot_col, sort=False)
    lot_to_indices = {lot: group.index.values for lot, group in lot_groups}

    print(f"  划分: train={train_end}, val={val_end-train_end}, test={n_test} ({len(test_lots_ordered)} lots)")

    # ==========================================
    # Baseline raw（OOM 自动降配）
    # ==========================================
    if is_mixed_mode:
        # Mixed mode: use lightweight fixed config (no OOM schedule)
        print(f"  [Baseline raw - Mixed Mode]...", end=" ", flush=True)
        t0 = time.time()
        try:
            model = create_model_with_params(
                MIXED_MODEL_CONFIG["n_estimators"], MIXED_MODEL_CONFIG["polynomial_features"]
            )
            model.fit(X_all_np[:val_end], y_all_np[:val_end])
            baseline_preds_raw = batched_predict(model, X_all_np[val_end:])
            baseline_cfg = {
                "n_estimators": MIXED_MODEL_CONFIG["n_estimators"],
                "poly_features": MIXED_MODEL_CONFIG["polynomial_features"],
            }
            baseline_time = time.time() - t0
            del model
            force_cleanup()
            print(f"{baseline_time:.0f}s | cfg={baseline_cfg}")
        except Exception as e:
            print(f"❌ Mixed mode baseline failed: {e}")
            force_cleanup()
            return None
    else:
        print(f"  [Baseline raw]...", end=" ", flush=True)
        t0 = time.time()
        try:
            baseline_preds_raw, baseline_cfg = fit_predict_with_oom_retry(
                X_all_np[:val_end], y_all_np[:val_end], X_all_np[val_end:],
                stage_name="baseline",
            )
            baseline_time = time.time() - t0
            print(f"{baseline_time:.0f}s | cfg={baseline_cfg}")
        except Exception as e:
            print(f"❌ {e}")
            force_cleanup()
            return None

    # ==========================================
    # Rolling raw (adaptive chunk + OOM 自动降配)
    # ==========================================
    if is_mixed_mode:
        print(f"  [Rolling raw] ⏭️ 混合模式下跳过 rolling（仅保留 baseline 结果）")
        rolling_preds_raw = np.full(n_total, np.nan)
        rolling_preds_raw[val_end:] = baseline_preds_raw
        rolling_time = 0.0
    else:
        print(f"  [Rolling raw adaptive] {len(test_lots_ordered)} lots...")
        rolling_preds_raw = np.full(n_total, np.nan)
        rolling_train_end = val_end

        total_lots = len(test_lots_ordered)
        rolling_start = time.time()

        rolling_chunk = min(max(ROLLING_CHUNK_LOTS, ROLLING_CHUNK_LOTS_MIN), ROLLING_CHUNK_LOTS_MAX)
        lot_ptr = 0
        chunk_i = 0

        while lot_ptr < total_lots:
            chunk_i += 1
            end_ptr = min(lot_ptr + rolling_chunk, total_lots)
            lot_chunk = test_lots_ordered[lot_ptr:end_ptr]

            chunk_test_indices = []
            for lot_id in lot_chunk:
                lot_indices = lot_to_indices[lot_id]
                lot_test_indices = lot_indices[lot_indices >= val_end]
                if len(lot_test_indices) > 0:
                    chunk_test_indices.append(lot_test_indices)

            if not chunk_test_indices:
                lot_ptr = end_ptr
                continue

            chunk_test_indices = np.unique(np.concatenate(chunk_test_indices))
            chunk_test_indices.sort()

            try:
                chunk_preds, used_cfg = fit_predict_with_oom_retry(
                    X_all_np[:rolling_train_end],
                    y_all_np[:rolling_train_end],
                    X_all_np[chunk_test_indices],
                    stage_name=f"rolling-chunk-{chunk_i}"
                )
                rolling_preds_raw[chunk_test_indices] = chunk_preds

                lot_max_idx = int(chunk_test_indices.max()) + 1
                if lot_max_idx > rolling_train_end:
                    rolling_train_end = lot_max_idx

                if rolling_chunk < ROLLING_CHUNK_LOTS_MAX:
                    rolling_chunk += 1

                lot_ptr = end_ptr

            except Exception as e:
                print(f"    [chunk {chunk_i}] ❌ {e}")
                force_cleanup()

                if rolling_chunk > ROLLING_CHUNK_LOTS_MIN:
                    rolling_chunk = max(ROLLING_CHUNK_LOTS_MIN, rolling_chunk // 2)
                    print(f"    [chunk {chunk_i}] ↘ 降低 rolling_chunk 到 {rolling_chunk} 后重试")
                    continue

                # 最小 chunk 仍失败则 fallback baseline，避免卡死
                rolling_preds_raw[chunk_test_indices] = baseline_preds_raw[chunk_test_indices - val_end]
                lot_max_idx = int(chunk_test_indices.max()) + 1
                if lot_max_idx > rolling_train_end:
                    rolling_train_end = lot_max_idx
                lot_ptr = end_ptr

            elapsed = time.time() - rolling_start
            done_lots = lot_ptr
            avg = elapsed / max(done_lots, 1)
            eta = avg * (total_lots - done_lots)
            print(f"    [chunk {chunk_i}] lots_done={done_lots}/{total_lots} ctx={rolling_train_end} "
                  f"| chunk={rolling_chunk} | {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

        rolling_time = time.time() - rolling_start
        print(f"  Rolling raw adaptive 完成: {rolling_time:.0f}s")

    # ==========================================
    # 评估（只看 raw）
    # ==========================================
    test_is_ref = df_meta.iloc[val_end:][SLOT_COL].isin(REFERENCE_SLOT_IDS).values
    test_is_nonref = ~test_is_ref
    y_test = y_all_np[val_end:]
    roll_raw = rolling_preds_raw[val_end:]
    n_nonref = test_is_nonref.sum()

    results = {
        "Baseline raw": compute_all_metrics(y_test[test_is_nonref], baseline_preds_raw[test_is_nonref]),
        "Rolling raw": compute_all_metrics(y_test[test_is_nonref], roll_raw[test_is_nonref]),
    }

    print(f"\n  📊 Non-ref ({n_nonref} 条) [RAW only]:")
    print(f"     {'方案':<16} {'MAE':>8} {'R²':>8} {'Acc@0.5':>8} {'Acc@1.0':>8}")
    print(f"     {'─'*52}")
    for name, r in results.items():
        if is_mixed_mode and name == "Rolling raw":
            continue  # Rolling == Baseline in mixed mode, skip duplicate line
        print(f"     {name:<16} {r['mae']:>8.4f} {r['r2']:>8.4f} "
              f"{r['acc05']:>7.1f}% {r['acc10']:>7.1f}%")

    # ── 混合模式：按 tool_name 拆分测试集指标 ──────────────────────
    per_tool_results = {}
    if TOOL_NAME_COL in df.columns:
        tool_labels_test = df[TOOL_NAME_COL].iloc[val_end:].values
        unique_tools = pd.Series(tool_labels_test).unique()
        if len(unique_tools) > 1:
            print(f"\n  📊 按 tool 拆分 (Non-ref) [RAW only]:")
            print(f"     {'tool':<20} {'方案':<16} {'MAE':>8} {'R²':>8} {'Acc@0.5':>8} {'Acc@1.0':>8}")
            print(f"     {'─'*76}")
            for tool in sorted(unique_tools):
                tool_mask = (tool_labels_test == tool) & test_is_nonref
                if tool_mask.sum() < 2:
                    continue
                t_results = {
                    "Baseline raw": compute_all_metrics(y_test[tool_mask], baseline_preds_raw[tool_mask]),
                    "Rolling raw":  compute_all_metrics(y_test[tool_mask], roll_raw[tool_mask]),
                }
                per_tool_results[tool] = t_results
                for name, r in t_results.items():
                    if is_mixed_mode and name == "Rolling raw":
                        continue  # Rolling == Baseline in mixed mode, skip duplicate line
                    print(f"     {str(tool):<20} {name:<16} {r['mae']:>8.4f} {r['r2']:>8.4f} "
                          f"{r['acc05']:>7.1f}% {r['acc10']:>7.1f}%")
    # ──────────────────────────────────────────────────────────────

    best_name = "Baseline raw" if is_mixed_mode else "Rolling raw"
    best = results[best_name]
    mode_label = "混合模式(Baseline)" if is_mixed_mode else "Rolling raw"
    print(f"\n  ✅ {mode_label} | MAE={best['mae']:.4f} R²={best['r2']:.4f} "
          f"Acc@0.5={best['acc05']:.1f}% Acc@1.0={best['acc10']:.1f}%")

    plot_path = plot_time_series_raw_only(
        y_test=y_test,
        baseline_raw=baseline_preds_raw,
        rolling_raw=roll_raw,
        test_is_ref=test_is_ref,
        results=results,
        dataset_name=dataset_name,
        output_dir=output_dir,
        is_mixed_mode=is_mixed_mode,
    )
    print(f"  📈 {plot_path}")

    del X_all_np, y_all_np, X_merged, X_pure, ref_features
    del baseline_preds_raw, rolling_preds_raw
    force_cleanup()

    return {
        "dataset": dataset_name,
        "n_rows": len(df),
        "n_lots": int(df[lot_col].nunique()),
        "n_features": int(len(selected_feature_cols)),
        "n_test": int(n_total - val_end),
        "n_nonref": int(n_nonref),
        "results": results,
        "per_tool_results": per_tool_results,
        "baseline_name": best_name,
        "baseline_time": float(baseline_time),
        "rolling_time": float(rolling_time),
    }


# ============================================================
# 混合推理：加载并合并所有 tool 数据
# ============================================================

def load_and_combine_all_tools(files):
    """
    加载所有文件并纵向拼接，使不同 tool 的 lot_id / wafer_id 保持唯一。
    若数据中没有 tool_name 列，则以文件名（不含扩展名）作为 tool 名。
    额外加入：每个 tool 限 lots、总行数抽样。
    """
    dfs = []
    for filepath in files:
        try:
            df = load_single_file(filepath)

            # 确保有 tool_name 列
            if TOOL_NAME_COL not in df.columns:
                tool_label = os.path.splitext(os.path.basename(filepath))[0]
                df[TOOL_NAME_COL] = tool_label
                print(f"  ⚠️  {os.path.basename(filepath)}: 无 '{TOOL_NAME_COL}' 列，使用文件名 '{tool_label}'")

            # 每个 tool 限 lots（可选）
            if LOT_COL in df.columns and MIXED_MAX_LOTS_PER_TOOL is not None:
                tmp = df.copy()
                if TIME_COL in tmp.columns:
                    tmp = tmp.sort_values(TIME_COL)
                keep_lots = tmp[LOT_COL].astype(str).drop_duplicates().head(MIXED_MAX_LOTS_PER_TOOL).tolist()
                df = df[df[LOT_COL].astype(str).isin(keep_lots)].copy()

            # 让 lot_id / wafer_id 在不同 tool 之间保持唯一
            if LOT_COL in df.columns:
                df[LOT_COL] = df[TOOL_NAME_COL].astype(str) + "_" + df[LOT_COL].astype(str)
            if WAFER_ID_COL in df.columns:
                df[WAFER_ID_COL] = df[TOOL_NAME_COL].astype(str) + "_" + df[WAFER_ID_COL].astype(str)

            print(f"  加载 {os.path.basename(filepath)}: {len(df)} 行 | tool={df[TOOL_NAME_COL].unique().tolist()}")
            dfs.append(df)  # 关键修复

        except Exception as e:
            print(f"  ❌ 加载失败 {os.path.basename(filepath)}: {e}")

    if not dfs:
        raise ValueError("没有成功加载任何数据文件")

    combined = pd.concat(dfs, ignore_index=True, sort=False)

    # 总行数太大则抽样
    if len(combined) > MIXED_MAX_ROWS:
        if TIME_COL in combined.columns:
            combined = combined.sample(MIXED_MAX_ROWS, random_state=42).sort_values(TIME_COL).reset_index(drop=True)
        else:
            combined = combined.sample(MIXED_MAX_ROWS, random_state=42).reset_index(drop=True)
        print(f"  ⚠️ mixed 行数过大，已抽样到 {len(combined)} 行")

    print(f"\n  混合数据汇总: {len(combined)} 行, "
          f"涉及 {combined[TOOL_NAME_COL].nunique()} 个 tool: "
          f"{combined[TOOL_NAME_COL].unique().tolist()}")
    return combined


def load_and_combine_all_tools_v2(files):
    """
    改进的混合数据加载（v2）：
    - 按 tool 分层采样
    - 保留完整的 lot 结构（不做行级采样）
    - 确保每个 tool 贡献平衡
    - 保留最新的 lots（时序相关性更强）
    """
    all_tool_data = []
    lot_col = LOT_COL  # Use global lot column name

    for filepath in files:
        try:
            df = load_single_file(filepath)

            # 确保有 tool_name 列
            if TOOL_NAME_COL not in df.columns:
                tool_label = os.path.splitext(os.path.basename(filepath))[0]
                df[TOOL_NAME_COL] = tool_label

            # 按时间排序
            if TIME_COL in df.columns:
                df = df.sort_values(TIME_COL)

            # ✅ 按 LOT 级别采样（不做行级采样）
            actual_lot_col = lot_col if lot_col in df.columns else (
                resolve_lot_column(df) if WAFER_ID_COL in df.columns else None
            )
            if actual_lot_col is None:
                print(f"  ⚠️ {os.path.basename(filepath)}: 无法确定 lot 列，跳过")
                continue

            unique_lots = df[actual_lot_col].unique()

            if MIXED_MAX_LOTS_PER_TOOL is not None and len(unique_lots) > MIXED_MAX_LOTS_PER_TOOL:
                # 保留最近的 lots（时序相关性）
                sampled_lots = unique_lots[-MIXED_MAX_LOTS_PER_TOOL:]
            else:
                sampled_lots = unique_lots

            tool_sampled = df[df[actual_lot_col].isin(sampled_lots)].copy()

            # 让 lot_id / wafer_id 在不同 tool 之间保持唯一
            tool_sampled[actual_lot_col] = (
                tool_sampled[TOOL_NAME_COL].astype(str) + "_" + tool_sampled[actual_lot_col].astype(str)
            )
            if WAFER_ID_COL in tool_sampled.columns and actual_lot_col != WAFER_ID_COL:
                tool_sampled[WAFER_ID_COL] = (
                    tool_sampled[TOOL_NAME_COL].astype(str) + "_" + tool_sampled[WAFER_ID_COL].astype(str)
                )

            all_tool_data.append(tool_sampled)
            print(f"  加载 {os.path.basename(filepath)}: {len(tool_sampled)} 行, {len(sampled_lots)} lots")

        except Exception as e:
            print(f"  ❌ 加载失败 {os.path.basename(filepath)}: {e}")

    if not all_tool_data:
        raise ValueError("没有成功加载任何数据文件")

    combined = pd.concat(all_tool_data, ignore_index=True, sort=False)

    # ✅ 仅当总行数超限时才做分层抽样（按 tool 分层，保留 lot 结构）
    if len(combined) > MIXED_MAX_ROWS:
        print(f"  ⚠️ 总行数 {len(combined)} > {MIXED_MAX_ROWS}，按 tool 分层抽样（保留 lot 结构）...")
        sampled_dfs = []
        tools = combined[TOOL_NAME_COL].unique()
        rows_per_tool = MIXED_MAX_ROWS // len(tools)

        for tool in tools:
            tool_df = combined[combined[TOOL_NAME_COL] == tool].copy()
            if len(tool_df) > rows_per_tool:
                # 按 lot 缩减（而非行级采样），保留最近的 lots
                tool_lots = tool_df[lot_col].unique() if lot_col in tool_df.columns else None
                if tool_lots is not None and len(tool_lots) > 1:
                    n_keep_lots = max(1, int(len(tool_lots) * rows_per_tool / len(tool_df)))
                    keep_lots = tool_lots[-n_keep_lots:]
                    tool_df = tool_df[tool_df[lot_col].isin(keep_lots)]
            sampled_dfs.append(tool_df)

        combined = pd.concat(sampled_dfs, ignore_index=True)
        if TIME_COL in combined.columns:
            combined = combined.sort_values(TIME_COL).reset_index(drop=True)

    actual_lot_col_in_combined = lot_col if lot_col in combined.columns else WAFER_ID_COL
    print(f"\n  ✅ 混合数据（保留 lot 结构）: {len(combined)} 行, "
          f"{combined[actual_lot_col_in_combined].nunique()} lots, "
          f"{combined[TOOL_NAME_COL].nunique()} tools")
    return combined


def run_mixed_pipeline(files):
    """
    将所有 tool 数据混合后跑一次完整 pipeline。
    tool_name 列会作为普通（类别）特征参与训练。
    run_pipeline 内部会自动按 tool_name 拆分测试集指标。
    """
    print(f"\n{'#'*70}")
    print(f"  🔀 混合推理 (Mixed Inference): 所有 tool 合并训练 & 推理")
    print(f"{'#'*70}")

    combined_df = load_and_combine_all_tools_v2(files)

    missing = [c for c in [TARGET_COL, SLOT_COL, TIME_COL] if c not in combined_df.columns]
    if missing:
        print(f"  ⚠️ 混合数据缺少列 {missing}，跳过混合推理")
        return None

    if len(combined_df) < 50:
        print(f"  ⚠️ 混合数据量太少 ({len(combined_df)} 行)，跳过")
        return None

    return run_pipeline(
        combined_df,
        dataset_name="[Mixed] ALL_TOOLS",
        output_dir=MIXED_OUTPUT_DIR,
        is_mixed_mode=True,
    )


# ============================================================
# 分层混合训练支持函数
# ============================================================

def _get_tool_name_from_file(filepath):
    """
    Extract tool name from file path using naming convention
    <TOOL_NAME>_<MMDD>_<MMDD>[.parquet|.csv] → <TOOL_NAME>.
    """
    stem = os.path.splitext(os.path.basename(filepath))[0]
    parts = stem.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else stem


def analyze_and_group_tools(mixed_result, all_results, single_metric_col="Rolling raw",
                             mixed_metric_col="Baseline raw", threshold=2.0):
    """
    Analyze per-tool MAE change from initial mixed training and classify tools into:
    - Beneficiary: MAE improved by > threshold%
    - Neutral: MAE change within ±threshold%
    - Degraded: MAE worsened by > threshold%

    Returns a dict with keys:
        'beneficiary', 'neutral', 'degraded',
        'mixed_group' (beneficiary + neutral),
        'single_group' (degraded),
        'analysis' (list of per-tool detail dicts).
    """
    per_tool = mixed_result.get("per_tool_results", {})

    single_by_tool = {}
    for r in all_results:
        for tn in r.get("tool_names", []):
            single_by_tool[tn] = r["results"].get(single_metric_col, {})

    beneficiary = []
    neutral = []
    degraded = []
    analysis = []

    for tool, t_res in per_tool.items():
        if tool not in single_by_tool:
            continue
        m_mix = t_res.get(mixed_metric_col, {})
        m_single = single_by_tool[tool]
        mixed_mae = m_mix.get("mae", float("nan"))
        single_mae = m_single.get("mae", float("nan"))
        if np.isnan(mixed_mae) or np.isnan(single_mae) or abs(single_mae) < 1e-9:
            continue

        mae_change_pct = (mixed_mae - single_mae) / single_mae * 100
        if mae_change_pct < -threshold:
            group = "beneficiary"
            beneficiary.append(tool)
        elif mae_change_pct > threshold:
            group = "degraded"
            degraded.append(tool)
        else:
            group = "neutral"
            neutral.append(tool)
        analysis.append({
            "tool": tool,
            "group": group,
            "mae_change_pct": mae_change_pct,
            "mixed_mae": mixed_mae,
            "single_mae": single_mae,
        })

    mixed_group = beneficiary + neutral
    single_group = degraded

    print(f"\n{'#'*70}")
    print(f"  🔍 工具分组分析（阈值 ±{threshold:.0f}%）")
    print(f"{'#'*70}")
    print(f"  ✅ 受益组 (beneficiary): {len(beneficiary)} 个 tools")
    for e in sorted([x for x in analysis if x["group"] == "beneficiary"], key=lambda x: x["mae_change_pct"]):
        print(f"     {str(e['tool']):<25} MAE 变化: {e['mae_change_pct']:>+7.1f}%")
    print(f"\n  ➡️  中性组 (neutral):    {len(neutral)} 个 tools")
    print(f"\n  ⚠️  退化组 (degraded):   {len(degraded)} 个 tools")
    for e in sorted([x for x in analysis if x["group"] == "degraded"], key=lambda x: x["mae_change_pct"], reverse=True):
        print(f"     {str(e['tool']):<25} MAE 变化: {e['mae_change_pct']:>+7.1f}%")
    print(f"\n  📋 策略:")
    print(f"     受益+中性 ({len(mixed_group)} tools) → 分层混合训练")
    print(f"     退化      ({len(single_group)} tools) → 保留单独训练结果")

    return {
        "beneficiary": beneficiary,
        "neutral": neutral,
        "degraded": degraded,
        "mixed_group": mixed_group,
        "single_group": single_group,
        "analysis": analysis,
    }


def run_stratified_mixed_training(files, group_tools):
    """
    Run mixed training on a filtered subset of tools (beneficiary + neutral).
    Degraded tools keep their single-tool training results instead.

    Args:
        files: All data file paths.
        group_tools: List of tool names to include in this stratified mixed training.

    Returns:
        Pipeline result dict, or None if insufficient data.
    """
    if not group_tools:
        print(f"  ⚠️ 分层混合组为空，跳过")
        return None

    filtered_files = [
        f for f in files
        if _get_tool_name_from_file(f) in group_tools
    ]

    if len(filtered_files) < 2:
        print(f"  ⚠️ 分层混合组文件数不足 ({len(filtered_files)})，需要至少 2 个，跳过")
        return None

    print(f"\n{'#'*70}")
    print(f"  🔀 分层混合推理 (Stratified Mixed): {len(filtered_files)} tools")
    for f in filtered_files:
        print(f"     {_get_tool_name_from_file(f)}")
    print(f"{'#'*70}")

    combined_df = load_and_combine_all_tools_v2(filtered_files)

    missing = [c for c in [TARGET_COL, SLOT_COL, TIME_COL] if c not in combined_df.columns]
    if missing:
        print(f"  ⚠️ 分层混合数据缺少列 {missing}，跳过")
        return None

    if len(combined_df) < 50:
        print(f"  ⚠️ 分层混合数据量太少 ({len(combined_df)} 行)，跳过")
        return None

    label = f"[Stratified-Mixed] {len(filtered_files)} tools"
    return run_pipeline(
        combined_df,
        dataset_name=label,
        output_dir=MIXED_OUTPUT_DIR,
        is_mixed_mode=True,
    )


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":

    print(f"{'#'*70}")
    print(f"  TabPFN VM Pipeline - Rolling Evaluation (RAW only)")
    print(f"{'#'*70}")
    print(f"\n配置: {FIXED_CONFIG}")
    print(f"多项式: {POLY_FEATURES}, batch_size: {PREDICT_BATCH_SIZE}, rolling_chunk_lots: {ROLLING_CHUNK_LOTS}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 显存: {total_mem:.1f} GB")

    files = discover_files(data_path)
    print(f"\n发现 {len(files)} 个数据文件:")
    for f in files:
        print(f"  {os.path.basename(f)}")

    completed = load_completed_datasets(RESUME_SUMMARY_PATH)
    if completed:
        print(f"\n断点续跑已启用：检测到 {len(completed)} 个已完成数据集（来自 {RESUME_SUMMARY_PATH}）")

    all_results = []
    total_start = time.time()

    if SKIP_SINGLE_TOOL_TRAINING:
        print(f"\n⏭️ SKIP_SINGLE_TOOL_TRAINING=True, 跳过单 tool 训练")
        if os.path.exists(RESUME_SUMMARY_PATH):
            existing_df = pd.read_csv(RESUME_SUMMARY_PATH)
            print(f"  加载已有结果: {len(existing_df)} 个数据集（来自 {RESUME_SUMMARY_PATH}）")
            for row in existing_df.to_dict('records'):
                dataset_name = str(row.get("dataset", ""))
                # Reconstruct compatible result dict from flat CSV columns
                stem = os.path.splitext(dataset_name)[0]
                # Infer tool name by stripping the trailing date-range suffix.
                # Expected filename format: <TOOL_NAME>_<MMDD>_<MMDD>[.parquet|.csv]
                # e.g. EPLBAB01_CHA1_1011_1229.parquet → tool_name = EPLBAB01_CHA1
                parts = stem.rsplit("_", 2)
                tool_name = parts[0] if len(parts) == 3 else stem
                all_results.append({
                    "dataset": dataset_name,
                    "n_rows": int(row.get("n_rows", 0)),
                    "n_lots": int(row.get("n_lots", 0)),
                    "n_features": int(row.get("n_features", 0)),
                    "n_test": int(row.get("n_test", 0)),
                    "n_nonref": int(row.get("n_nonref", 0)),
                    "results": {
                        "Rolling raw": {
                            "mae": float(row.get("mae", float("nan"))),
                            "r2": float(row.get("r2", float("nan"))),
                            "acc05": float(row.get("acc05", float("nan"))),
                            "acc10": float(row.get("acc10", float("nan"))),
                        },
                    },
                    "tool_names": [tool_name],
                    "per_tool_results": {},
                    "baseline_time": float(row.get("baseline_time", 0)),
                    "rolling_time": float(row.get("rolling_time", 0)),
                })
        else:
            print(f"  ⚠️ 未找到结果文件 {RESUME_SUMMARY_PATH}，无法跳过单 tool 训练")
    else:
        for file_i, filepath in enumerate(files):
            fname = os.path.basename(filepath)
            print(f"\n{'='*70}")
            print(f"  [{file_i+1}/{len(files)}] {fname}")
            print(f"{'='*70}")

            if ENABLE_RESUME and fname in completed:
                print("  ⏭️ 已在结果文件中，跳过")
                continue

            try:
                df = load_single_file(filepath)
                print(f"  加载: {len(df)} 行 × {df.shape[1]} 列")

                missing = [c for c in [TARGET_COL, SLOT_COL, TIME_COL] if c not in df.columns]
                if missing:
                    print(f"  ⚠️ 跳过: 缺少列 {missing}")
                    continue

                if len(df) < 50:
                    print(f"  ⚠️ 跳过: 数据量太少 ({len(df)} 行)")
                    continue

                result = run_pipeline(df, dataset_name=fname, is_mixed_mode=False)
                if result is not None:
                    if TOOL_NAME_COL in df.columns:
                        result["tool_names"] = df[TOOL_NAME_COL].unique().tolist()
                    all_results.append(result)
                    append_result_to_summary(RESUME_SUMMARY_PATH, result)
                    completed.add(fname)

                del df
                force_cleanup()

            except Exception as e:
                print(f"  ❌ 失败: {e}")
                import traceback
                traceback.print_exc()
                force_cleanup()

    total_time = time.time() - total_start
    if SKIP_SINGLE_TOOL_TRAINING:
        print(f"\n✅ 单 tool 结果已从文件加载（跳过训练）: {len(all_results)} 个数据集")
    else:
        print(f"\n✅ 单 tool 推理全部完成，总耗时: {total_time:.0f}s ({total_time/60:.1f} min)")

    # ============================================================
    # 混合推理（可选）
    # ============================================================
    mixed_result = None
    if RUN_MIXED_MODE and len(files) > 1:
        try:
            mixed_result = run_mixed_pipeline(files)
        except Exception as e:
            print(f"\n  ❌ 混合推理失败: {e}")
            import traceback
            traceback.print_exc()
            force_cleanup()
    elif RUN_MIXED_MODE and len(files) == 1:
        print(f"\n⚠️  仅发现 1 个数据文件，无需混合推理（与单 tool 推理相同）")

    # ============================================================
    # 最终对比汇总
    # ============================================================
    if all_results or mixed_result:
        print(f"\n{'#'*70}")
        print(f"  📋 最终对比汇总（单 tool 独立推理 vs 混合推理）")
        print(f"{'#'*70}")

        single_metric_col = "Rolling raw"
        mixed_metric_col = "Baseline raw"  # Mixed mode only runs baseline

        loaded_note = "（来自文件）" if SKIP_SINGLE_TOOL_TRAINING else ""
        print(f"\n  【单 tool 独立推理{loaded_note}】")
        print(f"  {'数据集':<30} {'MAE':>8} {'R²':>8} {'Acc@0.5':>8} {'Acc@1.0':>8}")
        print(f"  {'─'*60}")
        for r in all_results:
            m = r["results"].get(single_metric_col, {})
            print(f"  {r['dataset']:<30} {m.get('mae', float('nan')):>8.4f} "
                  f"{m.get('r2', float('nan')):>8.4f} "
                  f"{m.get('acc05', float('nan')):>7.1f}% "
                  f"{m.get('acc10', float('nan')):>7.1f}%")

        if mixed_result:
            print(f"\n  【混合推理（所有 tool 合并，仅 Baseline）】")
            print(f"  {'数据集':<30} {'MAE':>8} {'R²':>8} {'Acc@0.5':>8} {'Acc@1.0':>8}")
            print(f"  {'─'*60}")
            m = mixed_result["results"].get(mixed_metric_col, {})
            print(f"  {'[Mixed] ALL_TOOLS':<30} {m.get('mae', float('nan')):>8.4f} "
                  f"{m.get('r2', float('nan')):>8.4f} "
                  f"{m.get('acc05', float('nan')):>7.1f}% "
                  f"{m.get('acc10', float('nan')):>7.1f}%")

            per_tool = mixed_result.get("per_tool_results", {})
            if per_tool:
                print(f"\n  【混合推理 - 按 tool 拆分 vs 单 tool 独立推理对比】(混合:{mixed_metric_col} vs 单独:{single_metric_col})")
                print(f"  {'tool':<25} {'混合-MAE':>10} {'单独-MAE':>10} {'混合-R²':>8} {'单独-R²':>8}")
                print(f"  {'─'*65}")
                single_by_tool = {}
                for r in all_results:
                    for tn in r.get("tool_names", []):
                        single_by_tool[tn] = r["results"].get(single_metric_col, {})
                for tool, t_res in sorted(per_tool.items()):
                    m_mix = t_res.get(mixed_metric_col, {})
                    m_single = single_by_tool.get(tool, {})
                    print(f"  {str(tool):<25} "
                          f"{m_mix.get('mae', float('nan')):>10.4f} "
                          f"{m_single.get('mae', float('nan')):>10.4f} "
                          f"{m_mix.get('r2', float('nan')):>8.4f} "
                          f"{m_single.get('r2', float('nan')):>8.4f}")

            # ============================================================
            # 诊断分析：哪些 tool 改善了，哪些退化了
            # ============================================================
            if per_tool and single_by_tool:
                print(f"\n{'#'*70}")
                print(f"  📊 混合训练诊断分析")
                print(f"{'#'*70}")

                improvements = []
                degradations = []
                unchanged = []

                for tool, t_res in per_tool.items():
                    if tool not in single_by_tool:
                        continue
                    m_mix = t_res.get(mixed_metric_col, {})
                    m_single = single_by_tool[tool]
                    mixed_mae = m_mix.get("mae", float("nan"))
                    single_mae = m_single.get("mae", float("nan"))
                    if np.isnan(mixed_mae) or np.isnan(single_mae) or single_mae == 0:
                        continue
                    mae_change = (mixed_mae - single_mae) / single_mae * 100
                    entry = (tool, mae_change, mixed_mae, single_mae)
                    if mae_change < -2:
                        improvements.append(entry)
                    elif mae_change > 2:
                        degradations.append(entry)
                    else:
                        unchanged.append(entry)

                print(f"\n  ✅ 改善的 tools ({len(improvements)}):")
                if improvements:
                    print(f"     {'Tool':<25} {'变化%':>8} {'混合MAE':>10} {'单独MAE':>10}")
                    print(f"     {'─'*57}")
                    for tool, change, mixed_mae, single_mae in sorted(improvements, key=lambda x: x[1])[:10]:
                        print(f"     {str(tool):<25} {change:>7.1f}% {mixed_mae:>10.4f} {single_mae:>10.4f}")

                print(f"\n  ➡️  基本持平的 tools ({len(unchanged)}) (MAE 变化在 ±2% 内)")

                print(f"\n  ⚠️  退化的 tools ({len(degradations)}):")
                if degradations:
                    print(f"     {'Tool':<25} {'变化%':>8} {'混合MAE':>10} {'单独MAE':>10}")
                    print(f"     {'─'*57}")
                    for tool, change, mixed_mae, single_mae in sorted(degradations, key=lambda x: x[1], reverse=True)[:10]:
                        print(f"     {str(tool):<25} {change:>7.1f}% {mixed_mae:>10.4f} {single_mae:>10.4f}")

                all_entries = improvements + degradations + unchanged
                if all_entries:
                    avg_change = np.mean([x[1] for x in all_entries])
                    print(f"\n  📈 平均 MAE 变化: {avg_change:+.1f}%  "
                          f"(改善={len(improvements)}, 持平={len(unchanged)}, 退化={len(degradations)})")

    # ============================================================
    # 分层混合训练（基于初次混合训练结果自动分组后的优化策略）
    # ============================================================
    if RUN_MIXED_MODE and mixed_result is not None and all_results:
        per_tool_check = mixed_result.get("per_tool_results", {})
        if per_tool_check:
            try:
                groups = analyze_and_group_tools(
                    mixed_result, all_results,
                    single_metric_col="Rolling raw",
                    mixed_metric_col="Baseline raw",
                )
                # Only run stratified training when some tools degraded under full mixed training
                if groups["degraded"] and groups["mixed_group"]:
                    stratified_result = run_stratified_mixed_training(files, groups["mixed_group"])
                    if stratified_result is not None:
                        strat_per_tool = stratified_result.get("per_tool_results", {})
                        # Build single-tool lookup for comparison
                        single_by_tool_s = {}
                        for r in all_results:
                            for tn in r.get("tool_names", []):
                                single_by_tool_s[tn] = r["results"].get("Rolling raw", {})

                        print(f"\n{'#'*70}")
                        print(f"  📊 分层混合 vs 全量混合 vs 单独训练 对比")
                        print(f"{'#'*70}")
                        print(f"  {'tool':<25} {'分层MAE':>10} {'全量MAE':>10} {'单独MAE':>10} {'最优':>8}")
                        print(f"  {'─'*67}")

                        all_tool_names = set(per_tool_check.keys()) | set(strat_per_tool.keys()) | set(single_by_tool_s.keys())
                        strat_maes, full_mix_maes, single_maes_list = [], [], []

                        for tool in sorted(all_tool_names):
                            m_strat = strat_per_tool.get(tool, {}).get("Baseline raw", {})
                            m_full = per_tool_check.get(tool, {}).get("Baseline raw", {})
                            m_single = single_by_tool_s.get(tool, {})
                            strat_mae = m_strat.get("mae", float("nan"))
                            full_mae = m_full.get("mae", float("nan"))
                            single_mae = m_single.get("mae", float("nan"))

                            if not np.isnan(strat_mae):
                                strat_maes.append(strat_mae)
                            if not np.isnan(full_mae):
                                full_mix_maes.append(full_mae)
                            if not np.isnan(single_mae):
                                single_maes_list.append(single_mae)

                            candidates = {
                                "分层": strat_mae, "全量混合": full_mae, "单独": single_mae,
                            }
                            candidates = {k: v for k, v in candidates.items() if not np.isnan(v)}
                            best_label = min(candidates, key=candidates.get) if candidates else "N/A"
                            print(f"  {str(tool):<25} {strat_mae:>10.4f} {full_mae:>10.4f} "
                                  f"{single_mae:>10.4f} {best_label:>8}")

                        print(f"\n  💡 退化组 ({len(groups['degraded'])} tools) 保留单独训练结果:")
                        for tool in sorted(groups["degraded"]):
                            m_single = single_by_tool_s.get(tool, {})
                            print(f"     {str(tool):<25} 单独MAE={m_single.get('mae', float('nan')):>8.4f}")

                        # Effective combined MAE: stratified for mixed_group, single for degraded
                        effective_maes = []
                        for tool in sorted(all_tool_names):
                            if tool in groups["degraded"]:
                                m = single_by_tool_s.get(tool, {}).get("mae", float("nan"))
                            else:
                                m = strat_per_tool.get(tool, {}).get("Baseline raw", {}).get("mae", float("nan"))
                            if not np.isnan(m):
                                effective_maes.append(m)

                        if effective_maes and single_maes_list:
                            eff_avg = np.mean(effective_maes)
                            single_avg = np.mean(single_maes_list)
                            full_avg = np.mean(full_mix_maes) if full_mix_maes else float("nan")
                            print(f"\n  📈 综合对比（平均 MAE）:")
                            print(f"     单独训练:     {single_avg:.4f}")
                            print(f"     全量混合:     {full_avg:.4f}")
                            print(f"     分层混合策略: {eff_avg:.4f}")
                            if not np.isnan(single_avg) and abs(single_avg) > 1e-9:
                                change_vs_single = (eff_avg - single_avg) / single_avg * 100
                                print(f"     vs 单独训练: {change_vs_single:+.1f}%")
                else:
                    if not groups["degraded"]:
                        print(f"\n  ✅ 无退化 tool，无需分层训练（全量混合已是最优）")
                    else:
                        print(f"\n  ⚠️ 受益+中性组为空，所有 tool 均退化，跳过分层训练")
            except Exception as e:
                print(f"\n  ❌ 分层混合训练失败: {e}")
                import traceback
                traceback.print_exc()
                force_cleanup()

    grand_total = time.time() - total_start
    print(f"\n✅ 全部完成（含混合推理），总耗时: {grand_total:.0f}s ({grand_total/60:.1f} min)")
