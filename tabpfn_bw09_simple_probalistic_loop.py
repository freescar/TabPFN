#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
减量降本实验 (自包含): 参考片 + R2R下货值 -> 预测当站 BW09-2 EH OCD
-> 映射后道下货档 loop_count(2~9) -> 评估。

本版新增: lot-aware GroupKFold 评估 (--cv-mode groupkfold)
  - 按 lot 分组 K 折，消除同 lot 参考片跨集泄漏
  - 每折独立做特征工程 (LOO / PCA 仅在训练折拟合)
  - 聚合所有折的【非参考片】预测，在 pool 上算一次指标 (统计可信)
保留: holdout 单次时间切分 (--cv-mode holdout) 作对照

固化改动:
  1) 灵敏度 = 物理逐档档宽 (loop MET 区间中心之差)
  2) 诊断打印: loop 分布 / 划分 / 测试参考片占比
  3) 泄漏剔除: BW092EH_MET, rec1_value, BW092WETEH_MET, loop_count, 1050.030605_REC1
================================================================================
"""

import os
os.environ.setdefault("TABPFN_NO_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import gc
import re
import argparse
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_selection import f_regression
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")


# ============================================================
# 工艺常数
# ============================================================
FB_DC_TARGET1 = 81.0
PRE_OFFSET = 0.3127
REC1_GRADIENT = 0.1313
LOOP_OFFSET = 6.0

CURR_MET_TARGET = 76.0
POST_MET_TARGET = 81.0
POST_MET_UCL = 82.15
POST_MET_LCL = 79.84

RUN_VALUE_BOUNDS = np.array([0.0, 19.5, 26.2, 33.0, 39.8, 46.5, 53.5, 60.1, 100.0], dtype=np.float64)
CLASS_LABELS = np.arange(2, 10, dtype=int)

LOOP_MET_REGION = {
    2: (77.33915, 79.89950), 3: (76.45944, 77.33915), 4: (75.56660, 76.45944),
    5: (74.67376, 75.56660), 6: (73.79405, 74.67376), 7: (72.87495, 73.79405),
    8: (72.00837, 72.87495), 9: (66.76950, 72.00837),
}
LOOP_MET_WIDTH = {k: round(hi - lo, 5) for k, (lo, hi) in LOOP_MET_REGION.items()}
LOOP_MET_CENTER = {k: (lo + hi) / 2.0 for k, (lo, hi) in LOOP_MET_REGION.items()}

LEAKAGE_COLS = ["BW092EH_MET", "rec1_value", "BW092WETEH_MET", "loop_count", "1050.030605_REC1"]


# ============================================================
# 业务公式
# ============================================================
def ocd_to_run_value(ocd):
    return (FB_DC_TARGET1 - np.asarray(ocd, dtype=np.float64) - PRE_OFFSET) / REC1_GRADIENT - LOOP_OFFSET


def run_value_to_loop(rv, out_of_range="clip"):
    rv = np.asarray(rv, dtype=np.float64)
    idx = np.searchsorted(RUN_VALUE_BOUNDS[1:-1], rv, side="right")
    loop = CLASS_LABELS[np.clip(idx, 0, len(CLASS_LABELS) - 1)].astype(float)
    if out_of_range == "nan":
        loop[~((rv >= RUN_VALUE_BOUNDS[0]) & (rv < RUN_VALUE_BOUNDS[-1]))] = np.nan
    return loop


def ocd_to_loop(ocd, out_of_range="clip"):
    return run_value_to_loop(ocd_to_run_value(ocd), out_of_range=out_of_range)


def loop_center(loops):
    return np.array([LOOP_MET_CENTER[int(l)] for l in loops], dtype=np.float64)


# ============================================================
# IO / 工具
# ============================================================
def load_data(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def force_cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ============================================================
# 诊断
# ============================================================
def print_global_diagnostics(df, *, target_col, slot_col, lot_col, ref_slots, post_met_col):
    n = len(df)
    slots = df[slot_col].to_numpy()
    is_ref = np.isin(slots, ref_slots)
    print("\n----- 全量数据诊断 -----")
    print(f"  样本总数: {n}  | lot 数量: {df[lot_col].nunique()}  | 每 lot 平均: {n / df[lot_col].nunique():.2f} 片")
    loop_all = ocd_to_loop(pd.to_numeric(df[target_col], errors="coerce").to_numpy()).astype(int)
    vc = pd.Series(loop_all).value_counts().reindex(CLASS_LABELS, fill_value=0)
    print("  [全量 loop 分布] " + ", ".join(f"{k}:{int(vc[k])}" for k in CLASS_LABELS))
    print(f"  参考片占比 全量: {int(is_ref.sum())}/{n} ({is_ref.mean()*100:.1f}%)")
    print(f"  非参考片总数: {int((~is_ref).sum())}  (GroupKFold 后全部可进入测试 pool)")
    nonref_loop = loop_all[~is_ref]
    nvc = pd.Series(nonref_loop).value_counts().reindex(CLASS_LABELS, fill_value=0)
    print("  [非参考片 loop 分布] " + ", ".join(f"{k}:{int(nvc[k])}" for k in CLASS_LABELS))
    if post_met_col in df.columns:
        cov = pd.to_numeric(df[post_met_col], errors="coerce").notna().mean()
        print(f"  后道 MET 覆盖率: {cov*100:.1f}%")
    print("------------------------")


# ============================================================
# 参考片特征 (leave-one-out)
# ============================================================
def build_slot_ref_features(df, *, target_col, slot_col, lot_col, reference_slot_ids):
    slots = df[slot_col].to_numpy(dtype=np.int64)
    lots = df[lot_col].to_numpy()
    mets = df[target_col].to_numpy(dtype=np.float64)
    n = len(df)
    ref_ids = sorted(reference_slot_ids)
    is_ref = np.isin(slots, ref_ids)
    slot_min, slot_max = float(slots.min()), float(slots.max())
    slot_rng = max(slot_max - slot_min, 1.0)
    slot_norm = (slots - slot_min) / slot_rng

    feat = {
        "slot_id": slots.astype(np.float64), "slot_norm": slot_norm,
        "slot_center_dist": np.abs(slot_norm - 0.5), "slot_trend_sq": slot_norm ** 2,
        "is_ref_slot": is_ref.astype(np.float64),
    }
    if ref_ids:
        ref_arr = np.array(ref_ids, dtype=float)
        feat["nearest_ref_dist"] = np.min(np.abs(slots[:, None].astype(float) - ref_arr[None, :]), axis=1)
    else:
        feat["nearest_ref_dist"] = np.zeros(n)

    lot_mean = np.full(n, np.nan); lot_std = np.full(n, np.nan)
    lot_med = np.full(n, np.nan); lot_min = np.full(n, np.nan)
    lot_max = np.full(n, np.nan); lot_cnt = np.zeros(n); ref_interp = np.full(n, np.nan)
    ref_slot_met = {sid: np.full(n, np.nan) for sid in ref_ids}

    for lot in np.unique(lots):
        lm = lots == lot; lrm = lm & is_ref
        if lrm.sum() == 0:
            continue
        rslots = slots[lrm]; rmets = mets[lrm]
        smd = {int(s): float(m) for s, m in zip(rslots, rmets)}
        for sid in ref_ids:
            if sid in smd:
                ref_slot_met[sid][lm] = smd[sid]
        order = np.argsort(rslots)
        s_sorted = rslots[order].astype(float); m_sorted = rmets[order]
        tot = float(np.nansum(rmets)); tot2 = float(np.nansum(rmets ** 2)); nref = int(lrm.sum())

        nri = np.where(lm & ~is_ref)[0]
        if len(nri) > 0:
            mu = tot / nref; var = max(tot2 / nref - mu ** 2, 0.0)
            lot_mean[nri] = mu; lot_std[nri] = np.sqrt(var)
            lot_med[nri] = float(np.nanmedian(rmets))
            lot_min[nri] = float(np.nanmin(rmets)); lot_max[nri] = float(np.nanmax(rmets)); lot_cnt[nri] = nref
            for idx in nri:
                ref_interp[idx] = float(np.interp(slots[idx], s_sorted, m_sorted))

        for idx in np.where(lrm)[0]:
            cs = int(slots[idx]); own = smd.get(cs)
            if own is None or nref - 1 == 0:
                continue
            nloo = nref - 1
            loo_mets = np.array([m for s, m in smd.items() if s != cs], dtype=float)
            lot_mean[idx] = (tot - own) / nloo
            lot_std[idx] = np.sqrt(max((tot2 - own ** 2) / nloo - lot_mean[idx] ** 2, 0.0))
            lot_med[idx] = float(np.nanmedian(loo_mets))
            lot_min[idx] = float(np.nanmin(loo_mets)); lot_max[idx] = float(np.nanmax(loo_mets)); lot_cnt[idx] = nloo
            loo_items = sorted([(s, smd[s]) for s in smd if s != cs])
            if loo_items:
                ls = np.array([s for s, _ in loo_items], float); lv = np.array([m for _, m in loo_items], float)
                ref_interp[idx] = float(np.interp(float(cs), ls, lv))
            ref_slot_met[cs][idx] = np.nan

    feat.update({
        "lot_ref_met_mean": lot_mean, "lot_ref_met_std": lot_std, "lot_ref_met_median": lot_med,
        "lot_ref_met_min": lot_min, "lot_ref_met_max": lot_max, "lot_ref_met_range": lot_max - lot_min,
        "lot_ref_met_count": lot_cnt, "ref_met_interp": ref_interp,
    })
    for sid in ref_ids:
        feat[f"ref_slot_{sid}_met"] = ref_slot_met[sid]
        feat[f"ref_slot_{sid}_dev"] = ref_slot_met[sid] - lot_mean

    res = pd.DataFrame(feat, index=df.index)
    for c in res.columns:
        if res[c].isna().any():
            m = res[c].median()
            res[c] = res[c].fillna(0.0 if not np.isfinite(m) else m)
    return res.astype(np.float64)


# ============================================================
# R2R 下货值特征 (PCA 基由 fit_index 拟合)
# ============================================================
_REC_PAT = re.compile(r"_REC\d+$", re.IGNORECASE)


def discover_rec_columns(df, exclude):
    return [c for c in df.columns if c not in exclude and _REC_PAT.search(str(c))]


def parse_station(col):
    m = re.match(r"^(.*)_REC(\d+)$", str(col), re.IGNORECASE)
    return (m.group(1), int(m.group(2))) if m else (str(col), 0)


def build_rec_features(df, *, rec_cols, lot_col, slot_col, fit_mask, pca_components=4, interact_slot=True):
    """fit_mask: 布尔数组，PCA/标准化仅在 fit_mask=True 的行(训练折)上拟合。"""
    if not rec_cols:
        return pd.DataFrame(index=df.index)
    mat = df[rec_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    n, ncol = mat.shape
    feat = {}
    for j, c in enumerate(rec_cols):
        feat[f"rec_raw__{c}"] = mat[:, j]
    feat["rec_row_mean"] = np.nanmean(mat, axis=1)
    feat["rec_row_std"] = np.nanstd(mat, axis=1)
    feat["rec_row_min"] = np.nanmin(mat, axis=1)
    feat["rec_row_max"] = np.nanmax(mat, axis=1)
    feat["rec_row_range"] = feat["rec_row_max"] - feat["rec_row_min"]
    feat["rec_valid_count"] = np.sum(~np.isnan(mat), axis=1).astype(np.float64)

    st_map = {}
    for j, c in enumerate(rec_cols):
        st, _ = parse_station(c)
        st_map.setdefault(st, []).append(j)
    for st, idxs in st_map.items():
        sub = mat[:, idxs]; safe = re.sub(r"[^0-9A-Za-z]+", "_", st).strip("_")
        feat[f"rec_st_{safe}_mean"] = np.nanmean(sub, axis=1)
        if sub.shape[1] > 1:
            feat[f"rec_st_{safe}_std"] = np.nanstd(sub, axis=1)
            feat[f"rec_st_{safe}_delta"] = sub[:, -1] - sub[:, 0]

    lots = df[lot_col].to_numpy()
    dev = np.full_like(mat, np.nan)
    for lot in np.unique(lots):
        m = lots == lot
        dev[m] = mat[m] - np.nanmean(mat[m], axis=0)[None, :]
    feat["rec_lot_dev_absmean"] = np.nanmean(np.abs(dev), axis=1)
    feat["rec_lot_dev_l2"] = np.sqrt(np.nanmean(dev ** 2, axis=1))

    if ncol >= 1 and pca_components > 0:
        fit_idx = np.where(fit_mask)[0]
        tr = mat[fit_idx] if len(fit_idx) else mat
        cm = np.nan_to_num(np.nanmean(tr, axis=0))
        cs = np.nanstd(tr, axis=0); cs = np.where((cs > 1e-8) & np.isfinite(cs), cs, 1.0)
        std = (np.where(np.isnan(mat), cm[None, :], mat) - cm[None, :]) / cs[None, :]
        ncomp = min(int(pca_components), ncol, max(1, len(fit_idx)))
        try:
            ts = std[fit_idx] if len(fit_idx) else std
            tc = ts - np.nanmean(ts, axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(np.nan_to_num(tc), full_matrices=False)
            scores = np.nan_to_num(std) @ vh[:ncomp].T
            for k in range(scores.shape[1]):
                feat[f"rec_pca{k + 1}"] = scores[:, k]
        except np.linalg.LinAlgError:
            pass

    if interact_slot:
        slots = df[slot_col].to_numpy(dtype=np.float64)
        smin = float(np.nanmin(slots)); srng = max(float(np.nanmax(slots)) - smin, 1.0)
        sn = (slots - smin) / srng
        feat["rec_row_mean_x_slot"] = feat["rec_row_mean"] * sn
        if "rec_pca1" in feat:
            feat["rec_pca1_x_slot"] = feat["rec_pca1"] * sn

    res = pd.DataFrame(feat, index=df.index).replace([np.inf, -np.inf], np.nan)
    for c in res.columns:
        if res[c].isna().any():
            m = res[c].median()
            res[c] = res[c].fillna(0.0 if not np.isfinite(m) else m)
    return res.astype(np.float64)


# ============================================================
# 特征选择 / 清洗 / 泄漏剔除
# ============================================================
def select_features(X_train, y_train, X_all, *, max_features, max_missing_ratio, min_variance):
    cols = list(X_train.columns)
    if not cols:
        return X_all, cols
    miss = X_train.isna().mean()
    keep = miss[miss <= max_missing_ratio].index.tolist() or cols
    Xt = X_train[keep]
    num = Xt.select_dtypes(include=[np.number]).columns.tolist()
    if not num:
        sel = keep[:max_features]; return X_all[sel], sel
    Xn = Xt[num]; var = Xn.var(axis=0, skipna=True)
    keepn = var[var > min_variance].index.tolist() or num
    Xn = Xn[keepn]; Xf = Xn.fillna(Xn.median(numeric_only=True))
    try:
        sc, _ = f_regression(Xf, y_train)
        sc = np.nan_to_num(sc, nan=-1.0, posinf=-1.0, neginf=-1.0)
        order = np.argsort(sc)[::-1][:min(max_features, len(keepn))]
        sel = [keepn[i] for i in order]
    except Exception:
        sel = keepn[:max_features]
    return X_all[sel], sel


def coerce_numeric(X):
    X = X.copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c].dtype):
            X[c] = X[c].astype(np.float64)
    num = X.select_dtypes(include=[np.number]).columns
    X[num] = X[num].replace([np.inf, -np.inf], np.nan)
    return X


def drop_leakage(X, verbose=False):
    drop = [c for c in X.columns
            if str(c).replace("rec_raw__", "") in LEAKAGE_COLS or str(c) in LEAKAGE_COLS]
    if drop and verbose:
        print(f"  [泄漏剔除] {drop}")
    return X.drop(columns=drop) if drop else X


def residual_compensation(y_true, y_pred, lots, slots, reference_slot_ids):
    comp = y_pred.copy(); is_ref = np.isin(slots, reference_slot_ids)
    for lot in np.unique(lots):
        lm = lots == lot; rm = lm & is_ref; nm = lm & ~is_ref
        if rm.sum() == 0:
            continue
        bias = np.nanmean(y_true[rm] - y_pred[rm])
        if not np.isnan(bias):
            comp[nm] += bias
    return comp


# ============================================================
# TabPFN
# ============================================================
def make_model(model_path, n_estimators, softmax_temperature, poly_features, subsample_samples):
    from tabpfn import TabPFNRegressor
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabPFNRegressor(
        model_path=model_path, device=device, n_estimators=n_estimators,
        softmax_temperature=softmax_temperature, average_before_softmax=True,
        memory_saving_mode=True, ignore_pretraining_limits=True,
        inference_config={"SUBSAMPLE_SAMPLES": max(256, subsample_samples),
                          "POLYNOMIAL_FEATURES": max(1, poly_features)})


def predict_with_ci(model, X, ci_quantiles):
    r = model.predict(X, output_type="main", quantiles=ci_quantiles)
    return r["mean"], r["quantiles"][0], r["quantiles"][1]


# ============================================================
# 单折训练 + 预测 (给定 train_mask/test_mask)
# ============================================================
def run_one_split(df, args, ref_slots, train_mask, test_mask, *, verbose=False):
    n = len(df)
    y = df[args.target_col].astype(float).to_numpy(dtype=np.float64)
    slots = df[args.slot_col].to_numpy()
    lots = df[args.lot_col].astype(str).to_numpy()

    # 特征工程 (整表构造，但 PCA/标准化只在 train 行拟合；参考片 LOO 天然按 lot 局部)
    X = build_slot_ref_features(df, target_col=args.target_col, slot_col=args.slot_col,
                                lot_col=args.lot_col, reference_slot_ids=ref_slots)
    if args.use_rec_features:
        exclude = set(LEAKAGE_COLS) | {args.target_col, args.post_met_col, args.time_col,
                                       args.slot_col, args.lot_col, args.wafer_id_col,
                                       "GroundTruth", "is_reference"}
        rec_cols = discover_rec_columns(df, exclude)
        if rec_cols:
            X = pd.concat([X, build_rec_features(
                df, rec_cols=rec_cols, lot_col=args.lot_col, slot_col=args.slot_col,
                fit_mask=train_mask, pca_components=args.rec_pca_components,
                interact_slot=args.rec_interact_slot)], axis=1)

    X = coerce_numeric(X)
    X = drop_leakage(X, verbose=verbose)

    tr_idx = np.where(train_mask)[0]; te_idx = np.where(test_mask)[0]
    X_sel, sel = select_features(X.iloc[tr_idx], y[tr_idx], X,
                                 max_features=args.max_features,
                                 max_missing_ratio=args.max_missing_ratio,
                                 min_variance=args.min_variance)

    model = make_model(args.model_path, args.n_estimators, args.softmax_temperature,
                       args.poly_features, args.subsample_samples)
    model.fit(X_sel.iloc[tr_idx], y[tr_idx])
    y_pred_raw, q_lo_raw, q_hi_raw = predict_with_ci(
        model, X_sel.iloc[te_idx], [args.ci_quantile_lower, args.ci_quantile_upper])
    del model; force_cleanup()

    y_te = y[te_idx]; lots_te = lots[te_idx]; slots_te = slots[te_idx]
    y_pred = residual_compensation(y_te, y_pred_raw, lots_te, slots_te, ref_slots)
    bias = y_pred - y_pred_raw

    out = pd.DataFrame({
        "wafer_id": df[args.wafer_id_col].iloc[te_idx].to_numpy() if args.wafer_id_col in df.columns else te_idx,
        "lot_id": lots_te, "slot_id": slots_te,
        "is_ref": np.isin(slots_te, ref_slots),
        "ocd_true": y_te, "ocd_pred": y_pred,
        "ocd_ci_lo": q_lo_raw + bias, "ocd_ci_hi": q_hi_raw + bias,
    })
    if args.post_met_col in df.columns:
        out["post_met_true"] = pd.to_numeric(df[args.post_met_col].iloc[te_idx].to_numpy(), errors="coerce")
    return out, len(sel)


# ============================================================
# GroupKFold 主流程
# ============================================================
def run_groupkfold(df, args, ref_slots):
    df = df.sort_values(args.time_col).reset_index(drop=True)
    if args.lot_col not in df.columns:
        df[args.lot_col] = df[args.wafer_id_col].astype(str).str[:-2]
    groups = df[args.lot_col].astype(str).to_numpy()
    n = len(df)
    n_groups = len(np.unique(groups))
    n_folds = min(args.n_folds, n_groups)
    print(f"\n[CV] GroupKFold by lot: folds={n_folds}, lots={n_groups}")

    gkf = GroupKFold(n_splits=n_folds)
    fold_preds = []
    for fi, (tr, te) in enumerate(gkf.split(np.arange(n), groups=groups)):
        train_mask = np.zeros(n, bool); train_mask[tr] = True
        test_mask = np.zeros(n, bool); test_mask[te] = True
        slots_te = df[args.slot_col].to_numpy()[te]
        n_te_nonref = int((~np.isin(slots_te, ref_slots)).sum())
        out, n_feat = run_one_split(df, args, ref_slots, train_mask, test_mask, verbose=(fi == 0))
        fold_preds.append(out)
        print(f"  fold {fi+1}/{n_folds}: train={len(tr)} test={len(te)} "
              f"test_nonref={n_te_nonref} feats={n_feat}")
    pool = pd.concat(fold_preds, ignore_index=True)
    print(f"[CV] 聚合预测 pool: {len(pool)} 片 (全部 wafer 各被预测一次)")
    return pool


def run_holdout(df, args, ref_slots):
    df = df.sort_values(args.time_col).reset_index(drop=True)
    if args.lot_col not in df.columns:
        df[args.lot_col] = df[args.wafer_id_col].astype(str).str[:-2]
    n = len(df); val_end = int(n * args.val_ratio)
    train_mask = np.zeros(n, bool); train_mask[:val_end] = True
    test_mask = ~train_mask
    out, n_feat = run_one_split(df, args, ref_slots, train_mask, test_mask, verbose=True)
    print(f"  holdout: train={val_end} test={n-val_end} feats={n_feat}")
    return out


# ============================================================
# 灵敏度
# ============================================================
def closedloop_diagnostic_slope(df, *, post_met_col, target_col):
    d = df.copy()
    d["__loop"] = ocd_to_loop(pd.to_numeric(d[target_col], errors="coerce").to_numpy())
    sub = d[["__loop", post_met_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 5 or sub["__loop"].nunique() < 2:
        return {"slope": np.nan, "r": np.nan, "n": int(len(sub))}
    x = sub["__loop"].to_numpy(float); ym = sub[post_met_col].to_numpy(float)
    b, a = np.polyfit(x, ym, 1); yhat = a + b * x
    ss_res = float(np.sum((ym - yhat) ** 2)); ss_tot = float(np.sum((ym - ym.mean()) ** 2)) + 1e-12
    return {"slope": float(b), "r": float(np.sign(b) * np.sqrt(max(0.0, 1 - ss_res / ss_tot))), "n": int(len(sub))}


# ============================================================
# 评估
# ============================================================
def evaluate(pred_df, *, out_of_range="clip"):
    ev = pred_df[~pred_df["is_ref"].astype(bool)].copy()
    if ev.empty:
        raise ValueError("无非参考片，无法评估")
    true_loop = ocd_to_loop(ev["ocd_true"].to_numpy(), out_of_range).astype(int)
    pred_loop = ocd_to_loop(ev["ocd_pred"].to_numpy(), out_of_range).astype(int)
    ev["true_loop"], ev["pred_loop"] = true_loop, pred_loop
    ad = np.abs(pred_loop - true_loop)
    metric1 = {
        "n": int(len(ev)),
        "acc_within0": float(np.mean(ad == 0) * 100),
        "acc_within1": float(np.mean(ad <= 1) * 100),
        "mae_loop": float(np.mean(ad)),
        "severe_ge2_pct": float(np.mean(ad >= 2) * 100),
        "ocd_mae": float(np.mean(np.abs(ev["ocd_true"] - ev["ocd_pred"]))),
        "ocd_rmse": float(np.sqrt(np.mean((ev["ocd_true"] - ev["ocd_pred"]) ** 2))),
    }
    if "post_met_true" in ev.columns and ev["post_met_true"].notna().any():
        base = ev["post_met_true"].to_numpy(float).copy()
        base[np.isnan(base)] = POST_MET_TARGET
    else:
        base = np.full(len(ev), POST_MET_TARGET, dtype=float)
    shift = loop_center(pred_loop) - loop_center(true_loop)
    pred_met = base + shift
    ooc_true = (base > POST_MET_UCL) | (base < POST_MET_LCL)
    ooc_pred = (pred_met > POST_MET_UCL) | (pred_met < POST_MET_LCL)
    dev_true = np.abs(base - POST_MET_TARGET); dev_pred = np.abs(pred_met - POST_MET_TARGET)
    metric2 = {
        "post_target": POST_MET_TARGET, "ucl": POST_MET_UCL, "lcl": POST_MET_LCL,
        "p_ooc_pred": float(np.mean(ooc_pred) * 100), "p_ooc_true": float(np.mean(ooc_true) * 100),
        "extra_ooc_risk": float((np.mean(ooc_pred) - np.mean(ooc_true)) * 100),
        "p_pred_worse": float(np.mean(dev_pred > dev_true) * 100),
        "mean_abs_dev_pred": float(np.mean(dev_pred)), "mean_abs_dev_true": float(np.mean(dev_true)),
        "mean_abs_shift": float(np.mean(np.abs(shift))),
    }
    ev["post_met_base"] = base; ev["post_met_pred"] = pred_met
    ev["met_shift"] = shift; ev["ooc_pred"] = ooc_pred
    return {"metric1": metric1, "metric2": metric2, "pred_df": ev}


def per_class_acc(ev):
    print("  [按真实 loop 分档准确率]")
    for k in CLASS_LABELS:
        m = ev["true_loop"] == k
        nk = int(m.sum())
        if nk == 0:
            continue
        ad = np.abs(ev.loc[m, "pred_loop"] - ev.loc[m, "true_loop"])
        print(f"     loop {k}: n={nk:4d}  within0={np.mean(ad==0)*100:5.1f}%  within1={np.mean(ad<=1)*100:5.1f}%")


# ============================================================
# 绘图
# ============================================================
def make_plots(ev, out_dir):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(ev["true_loop"], ev["pred_loop"], labels=CLASS_LABELS)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set(xticks=range(len(CLASS_LABELS)), yticks=range(len(CLASS_LABELS)),
           xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
           xlabel="pred loop", ylabel="true loop", title="Loop Confusion Matrix (CV pool)")
    for i in range(len(CLASS_LABELS)):
        for j in range(len(CLASS_LABELS)):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loop_confusion_matrix.png"), dpi=120); plt.close()

    ev2 = ev.sort_values("ocd_true").reset_index(drop=True)
    x = np.arange(len(ev2))
    plt.figure(figsize=(16, 5))
    plt.plot(x, ev2["ocd_true"].to_numpy(), "k.", ms=4, label="true OCD")
    plt.plot(x, ev2["ocd_pred"].to_numpy(), color="steelblue", lw=0.8, alpha=0.8, label="pred OCD")
    plt.axhline(CURR_MET_TARGET, color="red", ls="--", lw=1, label=f"Target={CURR_MET_TARGET}")
    plt.title("BW09-2 EH OCD: true vs pred (CV pool, sorted by true)")
    plt.xlabel("wafer (sorted)"); plt.ylabel("OCD"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "ocd_true_vs_pred.png"), dpi=120); plt.close()


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser("减量降本实验 v3: lot-aware GroupKFold 评估")
    p.add_argument("--data-path", default="/ossfs/workspace/tools/A2_DBJOA_BW09_PLUS_20260101_20260601_merge_curr_pre_r2r_post_36tool.csv")
    p.add_argument("--output-dir", default="./results/tmp")
    p.add_argument("--target-col", default="BW092EH_MET")
    p.add_argument("--post-met-col", default="BW092WETEH_MET")
    p.add_argument("--time-col", default="start_time")
    p.add_argument("--slot-col", default="slot_id")
    p.add_argument("--lot-col", default="lot_id")
    p.add_argument("--wafer-id-col", default="wafer_id")
    p.add_argument("--reference-slot-ids", default="2,3,4,5,12,13,20,21,22,23")
    p.add_argument("--cv-mode", choices=["groupkfold", "holdout"], default="groupkfold")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.8)
    p.add_argument("--model-path",
                   default="/ossfs/workspace/xrfm/TabPFN-main/models/tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt")
    p.add_argument("--n-estimators", type=int, default=4)
    p.add_argument("--softmax-temperature", type=float, default=0.9)
    p.add_argument("--poly-features", type=int, default=1)
    p.add_argument("--subsample-samples", type=int, default=2048)
    p.add_argument("--max-features", type=int, default=120)
    p.add_argument("--max-missing-ratio", type=float, default=0.60)
    p.add_argument("--min-variance", type=float, default=1e-10)
    p.add_argument("--ci-quantile-lower", type=float, default=0.1)
    p.add_argument("--ci-quantile-upper", type=float, default=0.9)
    p.add_argument("--rec-pca-components", type=int, default=4)
    p.add_argument("--no-rec-features", dest="use_rec_features", action="store_false", default=True)
    p.add_argument("--no-rec-interact-slot", dest="rec_interact_slot", action="store_false", default=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ref_slots = [int(x) for x in args.reference_slot_ids.split(",") if x.strip()]

    df = load_data(args.data_path)
    print(f"[INFO] data shape={df.shape}  回归目标={args.target_col}  cv_mode={args.cv_mode}")
    print(f"[INFO] 工艺常数: 当站Target={CURR_MET_TARGET} | 后道Target={POST_MET_TARGET} "
          f"UCL={POST_MET_UCL} LCL={POST_MET_LCL}")
    print(f"[INFO] 泄漏剔除列: {LEAKAGE_COLS}")

    df_sorted = df.sort_values(args.time_col).reset_index(drop=True)
    lot_for_diag = args.lot_col if args.lot_col in df_sorted.columns else args.wafer_id_col
    print_global_diagnostics(df_sorted, target_col=args.target_col, slot_col=args.slot_col,
                             lot_col=lot_for_diag, ref_slots=ref_slots, post_met_col=args.post_met_col)

    # 灵敏度 (物理档宽)
    print("\n[灵敏度] 物理逐档档宽 (各档 MET 区间宽度):")
    for k in CLASS_LABELS:
        print(f"     loop {k}: width={LOOP_MET_WIDTH[k]:.5f}  center={LOOP_MET_CENTER[k]:.4f}")
    mid = np.mean([LOOP_MET_WIDTH[k] for k in [3, 4, 5, 6, 7, 8]])
    print(f"  中间档(3~8)平均宽度 ≈ {mid:.4f} nm/档 (对照工艺员 ≈1nm/档)")
    diag = closedloop_diagnostic_slope(df_sorted, post_met_col=args.post_met_col, target_col=args.target_col)
    print(f"  [闭环诊断] 数据回归 slope={diag['slope']:.4f} r={diag['r']:.4f} n={diag['n']} (≈0 正常, 不用于指标)")

    # 训练 + 预测
    if args.cv_mode == "groupkfold":
        pool = run_groupkfold(df, args, ref_slots)
    else:
        pool = run_holdout(df, args, ref_slots)

    # 评估 (pool 上算一次)
    print("\n[评估] 在聚合 pool 上计算指标 ...")
    out = evaluate(pool)
    m1, m2, ev = out["metric1"], out["metric2"], out["pred_df"]

    print("\n===== 指标1: 下货档准确率 (全量非参考片) =====")
    print(f"  n={m1['n']}  Acc@within0={m1['acc_within0']:.2f}%  Acc@within1={m1['acc_within1']:.2f}%")
    print(f"  MAE_loop={m1['mae_loop']:.3f}  Severe(|d|>=2)={m1['severe_ge2_pct']:.2f}%")
    print(f"  OCD_MAE={m1['ocd_mae']:.4f}  OCD_RMSE={m1['ocd_rmse']:.4f}")
    per_class_acc(ev)

    print("\n===== 指标2: 偏离 Target / OOC 概率 (物理档宽驱动) =====")
    print(f"  后道Target={m2['post_target']}  UCL={m2['ucl']}  LCL={m2['lcl']}")
    print(f"  预测下货 OOC 概率   P_ooc_pred   = {m2['p_ooc_pred']:.2f}%")
    print(f"  真实下货 OOC 概率   P_ooc_true   = {m2['p_ooc_true']:.2f}% (基线)")
    print(f"  额外 OOC 风险       extra_risk   = {m2['extra_ooc_risk']:.2f}%")
    print(f"  预测使偏离变大占比  P_pred_worse = {m2['p_pred_worse']:.2f}%")
    print(f"  平均|偏离Target|    预测={m2['mean_abs_dev_pred']:.4f} vs 真实={m2['mean_abs_dev_true']:.4f}")
    print(f"  平均注入 MET 偏移   |shift|       = {m2['mean_abs_shift']:.4f} nm")

    ev.to_csv(os.path.join(args.output_dir, "cv_pool_predictions.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([{**{f"m1_{k}": v for k, v in m1.items()},
                   **{f"m2_{k}": v for k, v in m2.items()}}]).to_csv(
        os.path.join(args.output_dir, "summary_metrics.csv"), index=False, encoding="utf-8-sig")
    try:
        make_plots(ev, args.output_dir)
    except Exception as e:
        print(f"  [WARN] 绘图失败: {e}")
    print(f"\n[OK] 结果输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
    print("Done.", flush=True)
