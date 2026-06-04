#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Constrained multi-task model for MET regression + loop classification.

This script implements:
1) Phase-1 monotonic feasibility analysis for REC discharge features.
2) Phase-2 constrained multi-task training with:
   - probabilistic regression head (Gaussian mean/std)
   - loop classification head
   - regression-classification consistency loss
   - physics monotonic constraint loss for selected REC features
3) Optional TabPFN baseline comparison on the same split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ.setdefault("TABPFN_NO_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

# Business constants aligned with existing scripts
FB_DC_TARGET1 = 81.0
PRE_OFFSET = 0.3127
REC1_GRADIENT = 0.1313
LOOP_OFFSET = 6.0
RUN_VALUE_BOUNDS = np.array([0.0, 19.5, 26.2, 33.0, 39.8, 46.5, 53.5, 60.1, 100.0], dtype=np.float64)
CLASS_LABELS = np.arange(2, 10, dtype=int)
LEAKAGE_COLS = ["BW092EH_MET", "rec1_value", "BW092WETEH_MET", "loop_count", "1050.030605_REC1"]


@dataclass
class MonotonicConstraint:
    feature: str
    feature_idx: int
    direction: float
    delta: float
    spearman: float


def ocd_to_run_value(ocd: np.ndarray) -> np.ndarray:
    ocd = np.asarray(ocd, dtype=np.float64)
    return (FB_DC_TARGET1 - ocd - PRE_OFFSET) / REC1_GRADIENT - LOOP_OFFSET


def run_value_to_loop(rv: np.ndarray, out_of_range: str = "clip") -> np.ndarray:
    rv = np.asarray(rv, dtype=np.float64)
    idx = np.searchsorted(RUN_VALUE_BOUNDS[1:-1], rv, side="right")
    loop = CLASS_LABELS[np.clip(idx, 0, len(CLASS_LABELS) - 1)].astype(float)
    if out_of_range == "nan":
        loop[~((rv >= RUN_VALUE_BOUNDS[0]) & (rv < RUN_VALUE_BOUNDS[-1]))] = np.nan
    return loop


def ocd_to_loop(ocd: np.ndarray, out_of_range: str = "clip") -> np.ndarray:
    return run_value_to_loop(ocd_to_run_value(ocd), out_of_range=out_of_range)


def run_value_to_met(rv: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, dtype=np.float64)
    return FB_DC_TARGET1 - PRE_OFFSET - REC1_GRADIENT * (rv + LOOP_OFFSET)


def get_loop_met_intervals() -> tuple[np.ndarray, np.ndarray]:
    # Map loop bins to MET intervals using run-value boundaries.
    lower = []
    upper = []
    for i in range(len(CLASS_LABELS)):
        rv_lo = RUN_VALUE_BOUNDS[i]
        rv_hi = RUN_VALUE_BOUNDS[i + 1]
        met_hi = run_value_to_met(rv_lo)
        met_lo = run_value_to_met(rv_hi)
        lower.append(min(met_lo, met_hi))
        upper.append(max(met_lo, met_hi))
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def ensure_lot_column(df: pd.DataFrame, lot_col: str, wafer_id_col: str) -> pd.DataFrame:
    out = df.copy()
    if lot_col in out.columns:
        out[lot_col] = out[lot_col].astype(str)
        return out
    if wafer_id_col in out.columns:
        out[lot_col] = out[wafer_id_col].astype(str).str[:-2]
        return out
    raise ValueError(f"Neither '{lot_col}' nor '{wafer_id_col}' is available")


def discover_rec_columns(df: pd.DataFrame, exclude_cols: set[str]) -> list[str]:
    pat = r"(^|_)REC\d*($|_)"
    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and pd.Series([c]).str.contains(pat, regex=True).iloc[0]:
            cols.append(c)
    return sorted(cols)


def robust_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 8:
        return float("nan")
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=np.float64)
    sx = float(np.std(xr))
    sy = float(np.std(yr))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def monotonic_feasibility_analysis(
    df_train: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: list[str],
    min_abs_spearman: float,
    delta_scale: float,
    min_delta: float,
    max_constraints: int,
) -> tuple[pd.DataFrame, list[MonotonicConstraint]]:
    rows: list[dict] = []
    constraints: list[MonotonicConstraint] = []

    y = pd.to_numeric(df_train[target_col], errors="coerce").to_numpy(dtype=np.float64)

    for c in feature_cols:
        x = pd.to_numeric(df_train[c], errors="coerce").to_numpy(dtype=np.float64)
        rho = robust_spearman(x, y)
        sigma = float(np.nanstd(x))
        if not np.isfinite(sigma):
            sigma = 0.0
        delta = max(min_delta, delta_scale * sigma)

        if np.isfinite(rho) and abs(rho) >= min_abs_spearman:
            direction = 1.0 if rho > 0 else -1.0
            usable = True
        else:
            direction = 0.0
            usable = False

        rows.append(
            {
                "feature": c,
                "spearman": rho,
                "abs_spearman": abs(rho) if np.isfinite(rho) else np.nan,
                "direction": int(direction),
                "delta": delta,
                "selected": bool(usable),
            }
        )

    analysis = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False, na_position="last")
    selected = analysis[analysis["selected"]].head(max(0, int(max_constraints)))
    for _, r in selected.iterrows():
        constraints.append(
            MonotonicConstraint(
                feature=str(r["feature"]),
                feature_idx=-1,  # fill later after final feature matrix is known
                direction=float(r["direction"]),
                delta=float(r["delta"]),
                spearman=float(r["spearman"]),
            )
        )
    return analysis, constraints


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    target_col: str,
    time_col: str,
    slot_col: str,
    lot_col: str,
    wafer_id_col: str,
    post_met_col: str,
    loop_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    exclude = {
        target_col,
        time_col,
        lot_col,
        wafer_id_col,
        post_met_col,
        loop_col,
        "GroundTruth",
        "is_reference",
    }
    exclude.update(LEAKAGE_COLS)

    num_cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if slot_col in df.columns and slot_col not in num_cols:
        num_cols.append(slot_col)

    X = df[num_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = float(np.nanmedian(X[c].to_numpy(dtype=np.float64)))
        if not np.isfinite(med):
            med = 0.0
        X[c] = X[c].fillna(med)

    return X.astype(np.float32), num_cols


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


class ConstrainedMultiTaskNet(torch.nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.reg_head = torch.nn.Linear(hidden_dim, 2)
        self.cls_head = torch.nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        reg_out = self.reg_head(h)
        mu = reg_out[:, 0]
        log_sigma = reg_out[:, 1].clamp(min=-6.0, max=2.5)
        logits = self.cls_head(h)
        return mu, log_sigma, logits


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gaussian_interval_probs(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    interval_lows: torch.Tensor,
    interval_highs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    # mu/sigma: [N], bounds: [C]
    z_lo = (interval_lows[None, :] - mu[:, None]) / sigma[:, None]
    z_hi = (interval_highs[None, :] - mu[:, None]) / sigma[:, None]
    p = normal_cdf(z_hi) - normal_cdf(z_lo)
    p = torch.clamp(p, min=eps)
    p = p / p.sum(dim=1, keepdim=True)
    return p


def evaluate_metrics(
    y_true_met: np.ndarray,
    y_pred_met: np.ndarray,
    y_true_loop: np.ndarray,
    y_pred_loop: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
) -> dict:
    abs_diff = np.abs(y_true_loop - y_pred_loop)
    return {
        "met_mae": float(mean_absolute_error(y_true_met, y_pred_met)),
        "met_rmse": float(np.sqrt(mean_squared_error(y_true_met, y_pred_met))),
        "met_r2": float(r2_score(y_true_met, y_pred_met)),
        "loop_acc": float(np.mean(y_true_loop == y_pred_loop) * 100.0),
        "loop_within1": float(np.mean(abs_diff <= 1) * 100.0),
        "loop_mae": float(np.mean(abs_diff)),
        "ci_width_mean": float(np.mean(q_hi - q_lo)),
        "ci_coverage_pct": float(np.mean((y_true_met >= q_lo) & (y_true_met <= q_hi)) * 100.0),
    }


def prepare_labels(df: pd.DataFrame, target_col: str, loop_col: str) -> tuple[np.ndarray, np.ndarray]:
    y_met = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(y_met)):
        raise ValueError(f"Target column '{target_col}' has non-numeric/NaN values")

    if loop_col in df.columns:
        loop_raw = pd.to_numeric(df[loop_col], errors="coerce").to_numpy(dtype=np.float64)
        valid_loop = np.isfinite(loop_raw) & np.isin(loop_raw.astype(int), CLASS_LABELS)
        y_loop = ocd_to_loop(y_met, out_of_range="clip").astype(int)
        y_loop[valid_loop] = loop_raw[valid_loop].astype(int)
    else:
        y_loop = ocd_to_loop(y_met, out_of_range="clip").astype(int)

    return y_met.astype(np.float32), y_loop.astype(int)


def train_constrained_model(
    X_train: np.ndarray,
    y_train_met: np.ndarray,
    y_train_loop: np.ndarray,
    X_test: np.ndarray,
    *,
    feature_names: list[str],
    constraints: list[MonotonicConstraint],
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    dropout: float,
    w_reg: float,
    w_cls: float,
    w_cons: float,
    w_mono: float,
    mono_margin: float,
    ci_q_low: float,
    ci_q_high: float,
    seed: int,
    device: str,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    feat_to_idx = {c: i for i, c in enumerate(feature_names)}
    active_constraints: list[MonotonicConstraint] = []
    for c in constraints:
        if c.feature in feat_to_idx:
            c.feature_idx = feat_to_idx[c.feature]
            active_constraints.append(c)

    model = ConstrainedMultiTaskNet(
        in_dim=X_train.shape[1],
        n_classes=len(CLASS_LABELS),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr = torch.from_numpy(X_train).float().to(device)
    ytr_met = torch.from_numpy(y_train_met).float().to(device)
    ytr_loop_idx = torch.from_numpy((y_train_loop - CLASS_LABELS[0]).astype(np.int64)).to(device)

    Xte = torch.from_numpy(X_test).float().to(device)

    lows_np, highs_np = get_loop_met_intervals()
    lows = torch.from_numpy(lows_np.astype(np.float32)).to(device)
    highs = torch.from_numpy(highs_np.astype(np.float32)).to(device)

    idx_all = np.arange(X_train.shape[0])

    for ep in range(epochs):
        model.train()
        np.random.shuffle(idx_all)

        for i in range(0, len(idx_all), batch_size):
            bidx_np = idx_all[i : i + batch_size]
            bidx = torch.from_numpy(bidx_np).to(device)
            xb = Xtr[bidx]
            yb_met = ytr_met[bidx]
            yb_loop = ytr_loop_idx[bidx]

            mu, log_sigma, logits = model(xb)
            sigma = torch.exp(log_sigma)
            var = sigma**2 + 1e-8

            reg_loss = 0.5 * (torch.log(var) + ((yb_met - mu) ** 2) / var).mean()
            cls_loss = F.cross_entropy(logits, yb_loop)

            probs_reg = gaussian_interval_probs(mu, sigma + 1e-6, lows, highs)
            probs_cls = torch.softmax(logits, dim=1)
            cons_loss = 0.5 * (
                F.kl_div(torch.log(probs_cls + 1e-8), probs_reg, reduction="batchmean")
                + F.kl_div(torch.log(probs_reg + 1e-8), probs_cls, reduction="batchmean")
            )

            mono_loss = torch.tensor(0.0, device=device)
            if active_constraints:
                losses = []
                for c in active_constraints:
                    xp = xb.clone()
                    xm = xb.clone()
                    xp[:, c.feature_idx] += c.delta
                    xm[:, c.feature_idx] -= c.delta
                    mu_p, _, _ = model(xp)
                    mu_m, _, _ = model(xm)
                    signed_diff = c.direction * (mu_p - mu_m)
                    losses.append(torch.relu(mono_margin - signed_diff).mean())
                mono_loss = torch.stack(losses).mean()

            total = w_reg * reg_loss + w_cls * cls_loss + w_cons * cons_loss + w_mono * mono_loss

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        if (ep + 1) % max(1, epochs // 5) == 0:
            print(
                f"  epoch {ep+1:4d}/{epochs} | reg={reg_loss.item():.4f} "
                f"cls={cls_loss.item():.4f} cons={cons_loss.item():.4f} mono={mono_loss.item():.4f}"
            )

    model.eval()
    with torch.no_grad():
        mu_te, log_sigma_te, logits_te = model(Xte)
        sigma_te = torch.exp(log_sigma_te)
        z_dist = torch.distributions.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        z_lo = z_dist.icdf(torch.tensor(ci_q_low, device=device))
        z_hi = z_dist.icdf(torch.tensor(ci_q_high, device=device))
        q_lo = (mu_te + sigma_te * z_lo).cpu().numpy()
        q_hi = (mu_te + sigma_te * z_hi).cpu().numpy()

        pred_met = mu_te.cpu().numpy()
        pred_loop = (torch.argmax(logits_te, dim=1) + CLASS_LABELS[0]).cpu().numpy()

    train_info = {
        "n_constraints_active": len(active_constraints),
        "active_constraints": [
            {
                "feature": c.feature,
                "direction": int(c.direction),
                "delta": c.delta,
                "spearman": c.spearman,
            }
            for c in active_constraints
        ],
    }

    return train_info, pred_met, pred_loop, q_lo, q_hi


def try_tabpfn_baseline(
    X_train: np.ndarray,
    y_train_met: np.ndarray,
    X_test: np.ndarray,
    *,
    ci_q_low: float,
    ci_q_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        from tabpfn import TabPFNRegressor
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] TabPFN not available: {exc}")
        return None

    model = TabPFNRegressor(device="cpu", n_estimators=4, ignore_pretraining_limits=True)
    model.fit(pd.DataFrame(X_train), y_train_met)
    out = model.predict(pd.DataFrame(X_test), output_type="main", quantiles=[ci_q_low, ci_q_high])
    return out["mean"], out["quantiles"][0], out["quantiles"][1]


def main() -> None:
    p = argparse.ArgumentParser("Constrained multi-task MET/loop trainer")
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", default="./results/constrained_multitask")
    p.add_argument("--target-col", default="BW092EH_MET")
    p.add_argument("--loop-col", default="loop_count")
    p.add_argument("--post-met-col", default="BW092WETEH_MET")
    p.add_argument("--time-col", default="start_time")
    p.add_argument("--slot-col", default="slot_id")
    p.add_argument("--lot-col", default="lot_id")
    p.add_argument("--wafer-id-col", default="wafer_id")
    p.add_argument("--val-ratio", type=float, default=0.8)

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--w-reg", type=float, default=1.0)
    p.add_argument("--w-cls", type=float, default=1.0)
    p.add_argument("--w-cons", type=float, default=0.4)
    p.add_argument("--w-mono", type=float, default=0.2)
    p.add_argument("--mono-margin", type=float, default=0.02)

    p.add_argument("--min-abs-spearman", type=float, default=0.08)
    p.add_argument("--delta-scale", type=float, default=0.05)
    p.add_argument("--min-delta", type=float, default=0.02)
    p.add_argument("--max-constraints", type=int, default=8)

    p.add_argument("--ci-quantile-lower", type=float, default=0.1)
    p.add_argument("--ci-quantile-upper", type=float, default=0.9)

    p.add_argument("--compare-tabpfn", action="store_true", default=False)

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_data(args.data_path)
    df = ensure_lot_column(df, args.lot_col, args.wafer_id_col)
    if args.time_col in df.columns:
        df = df.sort_values(args.time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    y_met, y_loop = prepare_labels(df, args.target_col, args.loop_col)
    X_df, feature_names = build_feature_matrix(
        df,
        target_col=args.target_col,
        time_col=args.time_col,
        slot_col=args.slot_col,
        lot_col=args.lot_col,
        wafer_id_col=args.wafer_id_col,
        post_met_col=args.post_met_col,
        loop_col=args.loop_col,
    )

    n = len(df)
    split = int(n * args.val_ratio)
    if split <= 0 or split >= n:
        raise ValueError("Invalid val-ratio; train/test split is empty")

    X = X_df.to_numpy(dtype=np.float32)
    X_train, X_test = X[:split], X[split:]
    X_train, X_test, x_mu, x_std = standardize_train_test(X_train, X_test)

    y_train_met, y_test_met = y_met[:split], y_met[split:]
    y_train_loop, y_test_loop = y_loop[:split], y_loop[split:]

    rec_cols = discover_rec_columns(
        df.iloc[:split],
        exclude_cols={args.target_col, args.time_col, args.lot_col, args.wafer_id_col, args.loop_col, args.post_met_col},
    )
    analysis_df, constraints = monotonic_feasibility_analysis(
        df.iloc[:split],
        target_col=args.target_col,
        feature_cols=rec_cols,
        min_abs_spearman=args.min_abs_spearman,
        delta_scale=args.delta_scale,
        min_delta=args.min_delta,
        max_constraints=args.max_constraints,
    )

    analysis_path = os.path.join(args.output_dir, "phase1_monotonic_analysis.csv")
    analysis_df.to_csv(analysis_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] phase-1 analysis saved: {analysis_path}")
    print(f"[INFO] selected constraints: {len(constraints)} / REC columns: {len(rec_cols)}")

    train_info, pred_met, pred_loop, q_lo, q_hi = train_constrained_model(
        X_train,
        y_train_met,
        y_train_loop,
        X_test,
        feature_names=feature_names,
        constraints=constraints,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        w_reg=args.w_reg,
        w_cls=args.w_cls,
        w_cons=args.w_cons,
        w_mono=args.w_mono,
        mono_margin=args.mono_margin,
        ci_q_low=args.ci_quantile_lower,
        ci_q_high=args.ci_quantile_upper,
        seed=args.seed,
        device=device,
    )

    metrics = evaluate_metrics(
        y_true_met=y_test_met,
        y_pred_met=pred_met,
        y_true_loop=y_test_loop,
        y_pred_loop=pred_loop,
        q_lo=q_lo,
        q_hi=q_hi,
    )

    print("\n===== constrained multitask metrics =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    baseline_metrics: dict | None = None
    if args.compare_tabpfn:
        base = try_tabpfn_baseline(
            X_train,
            y_train_met,
            X_test,
            ci_q_low=args.ci_quantile_lower,
            ci_q_high=args.ci_quantile_upper,
        )
        if base is not None:
            b_pred_met, b_q_lo, b_q_hi = base
            b_pred_loop = ocd_to_loop(b_pred_met, out_of_range="clip").astype(int)
            baseline_metrics = evaluate_metrics(
                y_true_met=y_test_met,
                y_pred_met=b_pred_met,
                y_true_loop=y_test_loop,
                y_pred_loop=b_pred_loop,
                q_lo=b_q_lo,
                q_hi=b_q_hi,
            )
            print("\n===== tabpfn baseline metrics =====")
            for k, v in baseline_metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")

    pred_out = pd.DataFrame(
        {
            "y_true_met": y_test_met,
            "y_pred_met": pred_met,
            "q_lo": q_lo,
            "q_hi": q_hi,
            "y_true_loop": y_test_loop,
            "y_pred_loop": pred_loop,
        }
    )
    pred_path = os.path.join(args.output_dir, "constrained_multitask_predictions.csv")
    pred_out.to_csv(pred_path, index=False, encoding="utf-8-sig")

    summary = {
        "n_rows": int(n),
        "n_train": int(split),
        "n_test": int(n - split),
        "feature_count": int(X_df.shape[1]),
        "device": device,
        "train_info": train_info,
        "metrics": metrics,
        "tabpfn_baseline_metrics": baseline_metrics,
        "files": {
            "phase1_monotonic_analysis": analysis_path,
            "predictions": pred_path,
        },
        "feature_stats": {
            "mean_shape": list(x_mu.shape),
            "std_shape": list(x_std.shape),
        },
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] summary saved: {summary_path}")


if __name__ == "__main__":
    main()
