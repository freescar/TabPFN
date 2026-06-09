from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("premet_postmet_twostage_tabpfn.py")


def _load_stage2_module() -> types.ModuleType:
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = type("Tensor", (), {})
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        synchronize=lambda: None,
    )
    sys.modules["torch"] = fake_torch

    fake_tabpfn = types.ModuleType("tabpfn")

    class _DummyRegressor:
        pass

    fake_tabpfn.TabPFNRegressor = _DummyRegressor
    sys.modules["tabpfn"] = fake_tabpfn

    spec = importlib.util.spec_from_file_location("stage2_module_for_tests", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage2_inference_does_not_require_true_postmet() -> None:
    mod = _load_stage2_module()

    n = 12
    df_test = pd.DataFrame(
        {
            "premet": np.linspace(1.0, 2.1, n),
            "tool": ["T1"] * n,
            "loop": [1] * n,
            "slot": [1] * n,
        }
    )
    stage1_result = {"df_test": df_test, "y_pred": np.linspace(1.1, 2.2, n)}

    infer_inputs: list[np.ndarray] = []

    def fake_train(*args, **kwargs):  # noqa: ANN002, ANN003
        return object(), object(), object(), 0.01

    def fake_infer(
        _model,
        _le_tool,
        _le_slot,
        _df_test,
        pre_met_values,
        **_kwargs,
    ):
        infer_inputs.append(pre_met_values.copy())
        return pre_met_values + 0.5

    mod.train_postmet_model = fake_train
    mod.infer_postmet = fake_infer

    result = mod.run_stage2_postmet(
        stage1_result=stage1_result,
        df_postmet_train=pd.DataFrame({"x": [1]}),
        dataset_name="demo",
        output_dir="/tmp",
        premet_col_in_test="premet",
        postmet_pre_met_col="premet",
        postmet_post_met_col="true_postmet",
        postmet_tool_col="tool",
        postmet_loop_count_col="loop",
        postmet_slot_col="slot",
        model_path="dummy",
        n_estimators=1,
        softmax_temperature=1.0,
        average_before_softmax=True,
        poly_features=1,
        subsample_samples=32,
    )

    assert result is not None
    assert len(infer_inputs) == 2
    np.testing.assert_allclose(infer_inputs[0], df_test["premet"].to_numpy(dtype=np.float32))
    np.testing.assert_allclose(infer_inputs[1], stage1_result["y_pred"].astype(np.float32))
    assert result["n_test"] == n
    assert result["n_eval"] == 0
    assert math.isnan(result["postmet_metrics_A"]["mae"])
    assert math.isnan(result["postmet_metrics_B"]["mae"])


def test_stage2_uses_true_postmet_only_for_eval_mask() -> None:
    mod = _load_stage2_module()

    n = 12
    y_true = np.array([10.0, 10.1, np.nan, 10.3, 10.4, np.nan, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1])
    df_test = pd.DataFrame(
        {
            "premet": np.linspace(5.0, 6.1, n),
            "tool": ["T1"] * n,
            "loop": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "slot": [1] * n,
            "true_postmet": y_true,
        }
    )
    stage1_result = {"df_test": df_test, "y_pred": np.linspace(4.8, 5.9, n)}

    infer_lengths: list[int] = []

    def fake_train(*args, **kwargs):  # noqa: ANN002, ANN003
        return object(), object(), object(), 0.01

    def fake_infer(
        _model,
        _le_tool,
        _le_slot,
        _df_test,
        pre_met_values,
        **_kwargs,
    ):
        infer_lengths.append(len(pre_met_values))
        return pre_met_values + 0.2

    mod.train_postmet_model = fake_train
    mod.infer_postmet = fake_infer
    mod.plot_postmet_timeseries = lambda **kwargs: None
    mod.plot_postmet_scatter = lambda **kwargs: None

    result = mod.run_stage2_postmet(
        stage1_result=stage1_result,
        df_postmet_train=pd.DataFrame({"x": [1]}),
        dataset_name="demo",
        output_dir="/tmp",
        premet_col_in_test="premet",
        postmet_pre_met_col="premet",
        postmet_post_met_col="true_postmet",
        postmet_tool_col="tool",
        postmet_loop_count_col="loop",
        postmet_slot_col="slot",
        model_path="dummy",
        n_estimators=1,
        softmax_temperature=1.0,
        average_before_softmax=True,
        poly_features=1,
        subsample_samples=32,
    )

    assert result is not None
    assert infer_lengths == [n, n]
    assert result["n_test"] == n
    assert result["n_eval"] == 10
    assert not math.isnan(result["postmet_metrics_A"]["mae"])
    assert not math.isnan(result["postmet_metrics_B"]["mae"])
