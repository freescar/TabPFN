#!/usr/bin/env python
"""
TabPFN 性能基准测试
目标：在不同行数（500-50000）、不同列数（100-2000）条件下，
测量单块GPU、单块CPU、两块CPU、四块CPU 的耗时和内存/显存占用。
"""

import os
import gc
import time
import json
import traceback
import numpy as np
import pandas as pd
import torch
import psutil
import threading

# ────────────────────────────────────────────────
# 实验矩阵
# ────────────────────────────────────────────────
ROW_SIZES    = [500, 1000, 5000, 10000, 50000]   # 样本数
COL_SIZES    = [100, 500, 1000, 2000]             # 特征数
N_ESTIMATORS = 4                                  # 减小以加速基准测试（可调）
PREDICT_ROWS = 200                                # 每次预测的样本数
MODEL_PATH   = "/ossfs/workspace/xrfm/TabPFN-main/models/tabpfn-v2.5-regressor-v2.5_default.ckpt"
OUTPUT_DIR   = "./results/benchmark"
OUTPUT_JSON  = os.path.join(OUTPUT_DIR, "tabpfn_benchmark.json")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "tabpfn_benchmark.csv")

# ────────────────────────────────────────────────
# 设备配置
# （TabPFN device 参数只接受字符串 "cpu" / "cuda" / "cuda:0" 等，
#   多核CPU通过 torch.set_num_threads 控制并行线程数）
# ────────────────────────────────────────────────
GPU_AVAILABLE = torch.cuda.is_available()

DEVICE_CONFIGS = []
if GPU_AVAILABLE:
    DEVICE_CONFIGS.append({"name": "GPU_1",  "device": "cuda", "cpu_threads": None})
DEVICE_CONFIGS.append(    {"name": "CPU_1",  "device": "cpu",  "cpu_threads": 1})
DEVICE_CONFIGS.append(    {"name": "CPU_2",  "device": "cpu",  "cpu_threads": 2})
DEVICE_CONFIGS.append(    {"name": "CPU_4",  "device": "cpu",  "cpu_threads": 4})


# ────────────────────────────────────────────────
# 内存监控工具
# ────────────────────────────────────────────────
class MemoryMonitor:
    """后台线程，每 0.1 s 采样一次 RAM / VRAM 峰值"""
    def __init__(self, device: str):
        self.device = device
        self.stop_event = threading.Event()
        self.peak_ram_mb  = 0.0
        self.peak_vram_mb = 0.0
        self._proc = psutil.Process(os.getpid())

    def _sample(self):
        ram = self._proc.memory_info().rss / 1024**2
        if ram > self.peak_ram_mb:
            self.peak_ram_mb = ram
        if self.device.startswith("cuda") and GPU_AVAILABLE:
            vram = torch.cuda.max_memory_allocated() / 1024**2
            if vram > self.peak_vram_mb:
                self.peak_vram_mb = vram

    def run(self):
        while not self.stop_event.is_set():
            self._sample()
            time.sleep(0.1)

    def start(self):
        if self.device.startswith("cuda") and GPU_AVAILABLE:
            torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_event.set()
        self._thread.join()
        self._sample()   # 最后再采一次


# ────────────────────────────────────────────────
# 合成数据生成
# ────────────────────────────────────────────────
def make_data(n_rows: int, n_cols: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    y = rng.standard_normal(n_rows).astype(np.float32)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_cols)])
    return X_df, y

# ────────────────────────────────────────────────
# 单次实验
# ────────────────────────────────────────────────
def run_single_experiment(n_rows, n_cols, device_cfg):
    from tabpfn import TabPFNRegressor   # 延迟导入避免全局污染

    device_name    = device_cfg["name"]
    device_str     = device_cfg["device"]
    cpu_threads    = device_cfg["cpu_threads"]

    print(f"\n  >> rows={n_rows}, cols={n_cols}, device={device_name}", flush=True)

    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

    # 设置 CPU 线程数
    original_threads = torch.get_num_threads()
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)

    # 生成数据
    X, y = make_data(n_rows, n_cols)
    X_train, y_train = X, y
    X_pred = X.iloc[:PREDICT_ROWS]

    # 启动内存监控
    monitor = MemoryMonitor(device_str)
    monitor.start()

    result = {
        "n_rows":         n_rows,
        "n_cols":         n_cols,
        "device":         device_name,
        "cpu_threads":    cpu_threads if cpu_threads else "N/A",
        "fit_time_s":     None,
        "predict_time_s": None,
        "total_time_s":   None,
        "peak_ram_mb":    None,
        "peak_vram_mb":   None,
        "error":          None,
    }

    try:
        model = TabPFNRegressor(
            model_path=MODEL_PATH,           # ← 指定本地模型路径
            device=device_str,
            n_estimators=N_ESTIMATORS,
            ignore_pretraining_limits=True,
            memory_saving_mode=True,
        )

        # ── fit ──
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t1 = time.perf_counter()
        result["fit_time_s"] = round(t1 - t0, 4)

        # ── predict ──
        t2 = time.perf_counter()
        _ = model.predict(X_pred)
        t3 = time.perf_counter()
        result["predict_time_s"] = round(t3 - t2, 4)
        result["total_time_s"]   = round(t3 - t0, 4)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"     ⚠️  错误: {result['error']}", flush=True)
        traceback.print_exc()

    finally:
        monitor.stop()
        result["peak_ram_mb"]  = round(monitor.peak_ram_mb,  2)
        result["peak_vram_mb"] = round(monitor.peak_vram_mb, 2)

        # 清理
        try:
            del model
        except Exception:
            pass
        gc.collect()
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()
        torch.set_num_threads(original_threads)

    print(
        f"     fit={result['fit_time_s']}s  "
        f"pred={result['predict_time_s']}s  "
        f"RAM={result['peak_ram_mb']}MB  "
        f"VRAM={result['peak_vram_mb']}MB",
        flush=True
    )
    return result


# ────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("TabPFN 性能基准测试")
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  行数: {ROW_SIZES}")
    print(f"  列数: {COL_SIZES}")
    print(f"  设备: {[d['name'] for d in DEVICE_CONFIGS]}")
    print(f"  n_estimators (基准): {N_ESTIMATORS}")
    print("=" * 65)

    # 启动前检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在，请检查路径: {MODEL_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    total = len(ROW_SIZES) * len(COL_SIZES) * len(DEVICE_CONFIGS)
    done  = 0

    for n_rows in ROW_SIZES:
        for n_cols in COL_SIZES:
            for dev_cfg in DEVICE_CONFIGS:
                done += 1
                print(f"\n[{done}/{total}]")
                rec = run_single_experiment(n_rows, n_cols, dev_cfg)
                all_results.append(rec)

                # 每次都保存（防止中途崩溃丢失数据）
                with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── 汇总打印 ──
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 65)
    print("基准测试结果汇总")
    print("=" * 65)

    # 按设备分组展示
    for dev in df["device"].unique():
        sub = df[df["device"] == dev].copy()
        print(f"\n▶ {dev}")
        pivot_fit  = sub.pivot(index="n_rows", columns="n_cols", values="fit_time_s")
        pivot_pred = sub.pivot(index="n_rows", columns="n_cols", values="predict_time_s")
        pivot_ram  = sub.pivot(index="n_rows", columns="n_cols", values="peak_ram_mb")
        print("  fit_time_s (行×列):\n", pivot_fit.to_string())
        print("  predict_time_s (行×列):\n", pivot_pred.to_string())
        print("  peak_ram_mb (行×列):\n", pivot_ram.to_string())
        if dev.startswith("GPU"):
            pivot_vram = sub.pivot(index="n_rows", columns="n_cols", values="peak_vram_mb")
            print("  peak_vram_mb (行×列):\n", pivot_vram.to_string())

    print(f"\n✅ JSON 结果: {OUTPUT_JSON}")
    print(f"✅ CSV  结果: {OUTPUT_CSV}")
    return all_results


if __name__ == "__main__":
    main()