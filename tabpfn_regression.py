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
# ★ 新增：屏蔽 matplotlib 缺字形警告（中文字符导致）
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font.*")

# ============================================================
# 配置区
# ============================================================

data_path = 'datasets/'

TARGET_COL = 'met'
TIME_COL = "start_time"
train_end_ratio = 0.7
val_end_ratio = 0.8

MAX_FEATURES = 1000
MODEL_PATH = './models/tabpfn-v2.5-regressor-v2.5_default.ckpt'

REFERENCE_SLOT_IDS = [2, 3, 4, 5, 12, 13, 20, 21, 22, 23]
SLOT_COL = "slot_id"
LOT_COL = "lot_id"
WAFER_ID_COL = "wafer_id"

OUTPUT_DIR = "./result/tool_all_fdc_1011_1229"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIXED_CONFIG = {
    "n_estimators": 32,
    "softmax_temperature": 0.5,
    "average_before_softmax": False,
}
POLY_FEATURES = 20
PREDICT_BATCH_SIZE = 200

# rolling chunk
ROLLING_CHUNK_LOTS = 8


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


def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 不 synchronize，避免频繁强制同步拖慢
        # torch.cuda.synchronize()


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
                              results, dataset_name, output_dir):
    safe_name = dataset_name.replace("/", "_").replace(" ", "_").replace(".", "_")
    n_test = len(y_test)
    x_axis = np.arange(n_test)
    test_is_nonref = ~test_is_ref

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for ax, (name, pred) in zip(axes, [
        ("Baseline raw", baseline_raw),
        ("Rolling raw", rolling_raw),
    ]):
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

def run_pipeline(df, dataset_name="dataset"):
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

    # ★ 仍然走纯数值输入（和你原 demo 一致）
    X_all_np = X_merged.select_dtypes(include=[np.number]).values.astype(np.float32)
    X_all_np = np.nan_to_num(X_all_np, nan=np.nan, posinf=np.nan, neginf=np.nan)
    y_all_np = y_series.values.astype(np.float32)

    test_lots_ordered = df_meta.iloc[val_end:][lot_col].unique()
    lot_groups = df_meta.groupby(lot_col, sort=False)
    lot_to_indices = {lot: group.index.values for lot, group in lot_groups}

    print(f"  划分: train={train_end}, val={val_end-train_end}, test={n_test} ({len(test_lots_ordered)} lots)")

    # ==========================================
    # Baseline raw
    # ==========================================
    print(f"  [Baseline raw]...", end=" ", flush=True)
    t0 = time.time()
    try:
        baseline_model = create_model()
        baseline_model.fit(X_all_np[:val_end], y_all_np[:val_end])
        baseline_preds_raw = batched_predict(baseline_model, X_all_np[val_end:])
        del baseline_model
        force_cleanup()
        baseline_time = time.time() - t0
        print(f"{baseline_time:.0f}s")
    except Exception as e:
        print(f"❌ {e}")
        force_cleanup()
        return None

    # ==========================================
    # Rolling raw (chunked)
    # ==========================================
    print(f"  [Rolling raw chunked] {len(test_lots_ordered)} lots, chunk={ROLLING_CHUNK_LOTS}...")
    rolling_preds_raw = np.full(n_total, np.nan)
    rolling_train_end = val_end

    total_lots = len(test_lots_ordered)
    rolling_start = time.time()

    lot_chunks = [
        test_lots_ordered[i:i + ROLLING_CHUNK_LOTS]
        for i in range(0, total_lots, ROLLING_CHUNK_LOTS)
    ]

    for chunk_i, lot_chunk in enumerate(lot_chunks, 1):
        chunk_test_indices = []
        for lot_id in lot_chunk:
            lot_indices = lot_to_indices[lot_id]
            lot_test_indices = lot_indices[lot_indices >= val_end]
            if len(lot_test_indices) > 0:
                chunk_test_indices.append(lot_test_indices)

        if not chunk_test_indices:
            continue

        chunk_test_indices = np.unique(np.concatenate(chunk_test_indices))
        chunk_test_indices.sort()

        try:
            model = create_model()
            model.fit(X_all_np[:rolling_train_end], y_all_np[:rolling_train_end])

            chunk_X = X_all_np[chunk_test_indices]
            chunk_preds = batched_predict(model, chunk_X)
            rolling_preds_raw[chunk_test_indices] = chunk_preds

            lot_max_idx = int(chunk_test_indices.max()) + 1
            if lot_max_idx > rolling_train_end:
                rolling_train_end = lot_max_idx

            del model
            force_cleanup()

        except Exception as e:
            print(f"    [chunk {chunk_i}/{len(lot_chunks)}] ❌ {e}")
            force_cleanup()
            rolling_preds_raw[chunk_test_indices] = baseline_preds_raw[chunk_test_indices - val_end]
            lot_max_idx = int(chunk_test_indices.max()) + 1
            if lot_max_idx > rolling_train_end:
                rolling_train_end = lot_max_idx

        elapsed = time.time() - rolling_start
        done_lots = min(chunk_i * ROLLING_CHUNK_LOTS, total_lots)
        avg = elapsed / max(done_lots, 1)
        eta = avg * (total_lots - done_lots)
        print(f"    [chunk {chunk_i}/{len(lot_chunks)}] lots_done={done_lots}/{total_lots} ctx={rolling_train_end} "
              f"| {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    rolling_time = time.time() - rolling_start
    print(f"  Rolling raw chunked 完成: {rolling_time:.0f}s")

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
        print(f"     {name:<16} {r['mae']:>8.4f} {r['r2']:>8.4f} "
              f"{r['acc05']:>7.1f}% {r['acc10']:>7.1f}%")

    # raw baseline = Rolling raw（你观察到普遍更好）
    best_name = "Rolling raw"
    best = results[best_name]
    print(f"\n  ✅ Baseline(更新): {best_name} | MAE={best['mae']:.4f} R²={best['r2']:.4f} "
          f"Acc@0.5={best['acc05']:.1f}% Acc@1.0={best['acc10']:.1f}%")

    plot_path = plot_time_series_raw_only(
        y_test=y_test,
        baseline_raw=baseline_preds_raw,
        rolling_raw=roll_raw,
        test_is_ref=test_is_ref,
        results=results,
        dataset_name=dataset_name,
        output_dir=OUTPUT_DIR,
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
        "baseline_name": best_name,
        "baseline_time": float(baseline_time),
        "rolling_time": float(rolling_time),
    }


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

    all_results = []
    total_start = time.time()

    for file_i, filepath in enumerate(files):
        fname = os.path.basename(filepath)
        print(f"\n{'='*70}")
        print(f"  [{file_i+1}/{len(files)}] {fname}")
        print(f"{'='*70}")

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

            result = run_pipeline(df, dataset_name=fname)
            if result is not None:
                all_results.append(result)

            del df
            force_cleanup()

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            force_cleanup()

    total_time = time.time() - total_start
    print(f"\n✅ 全部完成，总耗时: {total_time:.0f}s ({total_time/60:.1f} min)")
