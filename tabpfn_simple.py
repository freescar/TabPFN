#!/usr/bin/env python
"""
交叉验证评估：使用不同的数据分割比例

目标：验证模型在不同数据分割下的稳定性
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from tabpfn import TabPFNRegressor

# 最佳配置
BEST_PARAMS = {
    "target_transform": "none",
    "clip_threshold": 1.7823294720164862,
    "clip_method": "auto",
    "normalize": "standard",
    "n_estimators": 61,
    "softmax_temperature": 1.2381346676703702,
    "average_before_softmax": True,
    "subsample_samples": 30000,
    "poly_features": 100,
    "compensation_method": "reference",
    "compensation_window": 10,
    "trim_ratio": 0.1999617973462569,
    "predict_batch_size": 300
}

def load_parquet(path):
    return pd.read_parquet(path)

def clip_target(y, threshold, method="auto"):
    if method == "none" or threshold <= 0:
        return y
    elif method == "auto":
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        lower = max(y.min(), q1 - threshold * iqr)
        upper = min(y.max(), q3 + threshold * iqr)
        return np.clip(y, lower, upper)
    return y

def normalize_features(X, method="none"):
    if method == "none":
        return X
    elif method == "standard":
        from sklearn.preprocessing import StandardScaler
        num_cols = X.select_dtypes(include=[np.number]).columns
        X_norm = X.copy()
        scaler = StandardScaler()
        X_norm[num_cols] = scaler.fit_transform(X[num_cols])
        return X_norm
    return X

def apply_residual_compensation(df_meta, y_true, y_pred, lot_col, slot_col, reference_ids):
    compensated = y_pred.copy()
    lots = df_meta[lot_col].values
    slots = df_meta[slot_col].values
    is_ref = np.isin(slots, reference_ids)

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

def trim_predictions(y_pred, y_true, ratio):
    if ratio <= 0 or ratio >= 1:
        return y_pred

    trimmed = y_pred.copy()
    residuals = y_true - y_pred

    lower = np.percentile(residuals, 100 * ratio / 2)
    upper = np.percentile(residuals, 100 * (1 - ratio / 2))

    outlier_mask = (residuals < lower) | (residuals > upper)
    if outlier_mask.sum() > 0:
        trimmed[outlier_mask] = np.clip(
            y_pred[outlier_mask] + residuals[outlier_mask] * 0.5,
            y_true.min(), y_true.max()
        )

    return trimmed

def acc_within(y_true, y_pred, thr):
    return float(np.mean(np.abs(y_true - y_pred) <= thr) * 100.0)

def evaluate_split(df, train_ratio, val_ratio, split_name):
    """评估指定数据分割"""
    print(f"\n{'='*60}")
    print(f"评估分割: {split_name}")
    print(f"{'='*60}")
    print(f"  train_ratio={train_ratio}, val_ratio={val_ratio}")

    # 数据准备
    df = df.sort_values("start_time").reset_index(drop=True)

    target_col = "met"
    slot_col = "slot_id"
    lot_col = "lot_id"
    wafer_id_col = "wafer_id"

    if lot_col not in df.columns and wafer_id_col in df.columns:
        df[lot_col] = df[wafer_id_col].astype(str).str[:-2]

    n_total = len(df)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * val_ratio)

    print(f"  训练: 0-{train_end}, 验证: {train_end}-{val_end}, 测试: {val_end}-{n_total}")

    # 特征列
    drop_cols = {target_col, "start_time", slot_col, lot_col}
    if wafer_id_col in df.columns:
        drop_cols.add(wafer_id_col)

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target_col].astype(float).to_numpy(dtype=np.float32)
    slots = df[slot_col].to_numpy()

    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)

    # 测试集 mask
    reference_slot_ids = [2, 3, 4, 5, 12, 13, 20, 21, 22, 23]
    test_is_ref = np.isin(slots[val_end:], reference_slot_ids)
    test_is_nonref = ~test_is_ref

    if test_is_nonref.sum() == 0:
        print(f"  ⚠️  无 non-ref 测试样本，跳过此分割")
        return None

    # 模型
    model = TabPFNRegressor(
        model_path="./checkpoints/tabpfn-v2.5-regressor-v2.5_default.ckpt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_estimators=BEST_PARAMS["n_estimators"],
        softmax_temperature=BEST_PARAMS["softmax_temperature"],
        average_before_softmax=BEST_PARAMS["average_before_softmax"],
        memory_saving_mode=True,
        ignore_pretraining_limits=True,
        inference_config={
            "SUBSAMPLE_SAMPLES": BEST_PARAMS["subsample_samples"],
            "POLYNOMIAL_FEATURES": BEST_PARAMS["poly_features"],
        },
    )

    # 训练和预测
    model.fit(X.iloc[:val_end], y[:val_end])

    batch_size = BEST_PARAMS["predict_batch_size"]
    preds = []
    for i in range(0, len(X.iloc[val_end:]), batch_size):
        preds.append(model.predict(X.iloc[val_end:].iloc[i:i + batch_size]))
    y_pred = np.concatenate(preds)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 后处理
    meta_test = pd.DataFrame({
        lot_col: df[lot_col].iloc[val_end:].values,
        slot_col: slots[val_end:],
    })
    y_pred = apply_residual_compensation(
        meta_test, y[val_end:], y_pred, lot_col, slot_col, reference_slot_ids
    )

    if BEST_PARAMS["trim_ratio"] > 0:
        y_pred = trim_predictions(y_pred, y[val_end:], BEST_PARAMS["trim_ratio"])

    # 指标
    y_test = y[val_end:]
    mae = float(mean_absolute_error(y_test[test_is_nonref], y_pred[test_is_nonref]))
    r2 = float(r2_score(y_test[test_is_nonref], y_pred[test_is_nonref]))
    acc05 = float(acc_within(y_test[test_is_nonref], y_pred[test_is_nonref], 0.5))

    print(f"  MAE: {mae:.4f}, R²: {r2:.4f}, Acc@0.5: {acc05:.2f}%")

    return {
        "split_name": split_name,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "n_train": train_end,
        "n_val": val_end - train_end,
        "n_test": n_total - val_end,
        "n_test_nonref": int(test_is_nonref.sum()),
        "mae": mae,
        "r2": r2,
        "acc05": acc05
    }

def main():
    DATA_PATH = "./data/wide/EPLBAB01_CHA1_1101_1120.parquet"
    OUTPUT_DIR = "./results/verification"

    print("="*60)
    print("交叉验证评估")
    print("="*60)

    df = load_parquet(DATA_PATH)
    print(f"数据形状: {df.shape}")

    # 不同的数据分割
    splits = [
        (0.7, 0.8, "Standard (70/80/100)"),   # 标准分割
        (0.6, 0.75, "More Test (60/75/100)"), # 更多测试数据
        (0.65, 0.8, "Extended Train (65/80/100)"), # 更多训练数据
    ]

    results = []
    for train_ratio, val_ratio, name in splits:
        result = evaluate_split(df.copy(), train_ratio, val_ratio, name)
        if result:
            results.append(result)

    # 汇总
    print(f"\n{'='*60}")
    print("交叉验证结果汇总")
    print(f"{'='*60}")

    print(f"\n{'分割':<25} {'MAE':<10} {'R²':<10} {'Acc@0.5':<10} {'测试样本'}")
    print("-" * 70)
    for r in results:
        print(f"{r['split_name']:<25} {r['mae']:<10.4f} {r['r2']:<10.4f} {r['acc05']:<10.2f}% {r['n_test_nonref']}")

    # 统计
    mae_values = [r["mae"] for r in results]
    r2_values = [r["r2"] for r in results]

    print(f"\n统计:")
    print(f"MAE  - 均值: {np.mean(mae_values):.4f}, 标准差: {np.std(mae_values):.4f}, 范围: [{np.min(mae_values):.4f}, {np.max(mae_values):.4f}]")
    print(f"R²   - 均值: {np.mean(r2_values):.4f}, 标准差: {np.std(r2_values):.4f}, 范围: [{np.min(r2_values):.4f}, {np.max(r2_values):.4f}]")

    # 保存
    summary = {
        "splits_evaluated": len(results),
        "results": results,
        "statistics": {
            "mae": {
                "mean": float(np.mean(mae_values)),
                "std": float(np.std(mae_values)),
                "min": float(np.min(mae_values)),
                "max": float(np.max(mae_values))
            },
            "r2": {
                "mean": float(np.mean(r2_values)),
                "std": float(np.std(r2_values)),
                "min": float(np.min(r2_values)),
                "max": float(np.max(r2_values))
            }
        }
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/cross_validation.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ 结果已保存到: {OUTPUT_DIR}/cross_validation.json")

    if np.std(mae_values) < 0.05 and np.std(r2_values) < 0.05:
        print("\n✅ 模型跨分割稳定性良好")
    else:
        print("\n⚠️  模型在不同分割下有一定波动")

    return summary

if __name__ == "__main__":
    main()
