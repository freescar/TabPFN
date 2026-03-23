# TabPFN 技术汇报材料

---

## 一、TabPFN 是什么？

**TabPFN**（Tabular Prior-Data Fitted Network）是由弗莱堡大学、Prior Labs 等机构联合开发的**表格数据专用基础模型**，属于"先验拟合网络（PFN）"系列。其核心思想是：

> **不是在你的数据上训练模型，而是在"元学习"阶段训练一个能在无数合成数据集上做贝叶斯最优预测的 Transformer，从而实现零/极少样本下的高质量表格预测。**

从机器学习角度来说，TabPFN 实现了**上下文学习（In-Context Learning）**在表格数据上的应用——将训练集作为"prompt"喂给模型，模型一次前向传播即输出预测结果，无需梯度更新。

---

## 二、TabPFN 的发展历程

### 2.1 原始 TabPFN（v1，2022）

- 发表于 **ICLR 2023**（Hollmann et al.）
- 核心贡献：将 **Prior-Data Fitted Network** 应用于小型表格数据集
- **限制极大**：仅支持分类任务，最多 1000 样本、100 特征、10 类别
- 模型极轻量（约 100MB），但性能在小数据集上已超越 XGBoost
- 推理时间：CPU 上毫秒级（相当于直接做一次 Transformer 前向传播）

### 2.2 TabPFN v2（2024）

**论文**：*"TabPFN v2: Improved In-Context Learning for Tabular Data"*（Hollmann et al., 2024）

**重大突破**：

| 维度 | v1 | v2 |
|------|----|----|
| 任务类型 | 仅分类 | **分类 + 回归** |
| 最大样本数 | 1,000 | **10,000** |
| 最大特征数 | 100 | **500** |
| 最大类别数 | 10 | 10 |
| 模型架构 | 标准 Transformer | **Per-Feature Transformer** |

**关键技术创新**：

1. **Per-Feature Transformer**：每个特征×每个样本作为独立 token，特征维度和样本维度双重注意力机制
2. **回归头设计（BarDistribution）**：将回归问题转化为离散概率分布预测，可输出完整后验分布而非点估计
3. **大规模合成数据预训练**：引入更复杂的合成 prior（包括更真实的数据生成过程模拟）
4. **集成推理（Ensemble Inference）**：通过多次不同预处理/特征扰动的前向传播进行集成
5. **SVD 特征降维**：自动处理高维特征

### 2.3 TabPFN v2.5（2025年11月，包版本 6.0.0）

**技术报告**：*"TabPFN 2.5 Model Report"*（Prior Labs, 2025）

**关键升级**：

| 维度 | v2 | v2.5 |
|------|----|------|
| 最大样本数 | 10,000 | **50,000** |
| 最大特征数 | 500 | **2,000** |
| 预训练数据 | 纯合成 | **合成 + 真实数据混合** |
| 分类器权重 | 基础权重 | 在真实数据上微调 |
| 许可证 | Apache 2.0 | TABPFN-2.5 非商业许可 |

v2.5 是目前（截至 2026 年 3 月）的**默认版本**，安装最新 `tabpfn` 包（版本 ≥ 6.0.0）即可使用。

### 2.4 版本演进总结

```
2022          2024            2025-Nov        2026（当前）
  |              |                |               |
TabPFN v1    TabPFN v2       TabPFN v2.5    持续迭代
(分类/小数据) (分类+回归/中数据) (更大规模)   (微调/微调工具链)
```

---

## 三、TabPFN v2 技术细节

### 3.1 整体架构

TabPFN v2 的核心架构是 **PerFeatureTransformer**：

```
输入: [训练集 (N×D) + 测试集 (M×D)]
           ↓
   特征维度 × 样本维度 双重注意力
           ↓
  分类: Softmax 输出概率  /  回归: BarDistribution 输出分布
```

核心类 `PerFeatureTransformer`（`src/tabpfn/architectures/base/transformer.py`）：

- 每个"token"对应一个样本的一个特征（或一组特征）
- 两种注意力：**item attention**（样本间，捕捉"相似训练点"信息）和 **feature attention**（特征间，捕捉特征相关性）

### 3.2 模型配置（ModelConfig）

```python
# src/tabpfn/architectures/base/config.py
class ModelConfig:
    emsize: int = 192          # 嵌入维度
    features_per_group: int = 2 # 特征分组大小（用于 feature attention）
    nhead: int = 6              # 注意力头数
    nlayers: int = 12           # Transformer 层数
    nhid_factor: int = 4        # FFN 隐层扩展倍数（emsize × 4）
    feature_positional_embedding = "subspace"  # 特征位置编码
    nan_handling_enabled: bool = True  # 原生处理缺失值
```

### 3.3 回归的 BarDistribution 设计

这是 TabPFN v2 的关键创新之一。回归不输出单一预测值，而是输出一个**离散化的概率分布**（柱状分布），可以：

- 输出 **均值（mean）**、**中位数（median）**、**众数（mode）**
- 输出任意 **分位数（quantiles）**
- 获取完整**预测区间（prediction interval）**

这对虚拟量测非常有价值——不仅预测值，还能给出**预测不确定度**。

### 3.4 集成推理机制

v2 通过 `n_estimators` 个不同"prompt"进行集成，每次前向传播对输入数据做以下随机扰动：

- **特征位移（feature shift）**：随机打乱或旋转特征顺序（模拟特征位置无关性）
- **类别位移（class shift）**：分类时随机打乱类别顺序（模拟类别无关性）
- **不同预处理变换**：轮流使用不同的数值变换（分位数变换、幂变换等）
- **子采样（subsampling）**：大数据集下随机采样子集

### 3.5 预处理流水线

v2 的两套预处理配置（默认交替使用）：

**分类预处理（v2）**：

```python
# 预处理1：分位数均匀粗粒化 + SVD 降维 + 常见类别有序编码
PreprocessorConfig("quantile_uni_coarse",
                   append_original="auto",
                   categorical_name="ordinal_very_common_categories_shuffled",
                   global_transformer_name="svd")
# 预处理2：不做数值变换 + 类别当数值处理
PreprocessorConfig("none", categorical_name="numeric")
```

**回归预处理（v2）**：

```python
# 预处理1：分位数均匀变换 + 原始特征拼接 + SVD
PreprocessorConfig("quantile_uni", append_original=True,
                   categorical_name="ordinal_very_common_categories_shuffled",
                   global_transformer_name="svd")
# 预处理2：幂变换 + One-Hot 编码
PreprocessorConfig("safepower", categorical_name="onehot")
```

---

## 四、TabPFN v2.5 技术细节

### 4.1 相比 v2 的核心变化

**1. 规模突破：更大的预训练极限**

```python
# v2 的配置
MAX_NUMBER_OF_FEATURES = 500
MAX_NUMBER_OF_SAMPLES = 10_000

# v2.5 的配置
MAX_NUMBER_OF_FEATURES = 2000
MAX_NUMBER_OF_SAMPLES = 50_000
```

**2. 特征超过 500 时的子采样策略**

当特征数 > 500 时，v2.5 每个 estimator 随机采样 500 个特征，因此需要用**较大的 `n_estimators`**（如 16~32）才能充分覆盖所有特征。

**3. 新预处理配置（v2.5 专用）**

```python
# v2.5 分类预处理（squashing_scaler + svd_quarter_components）
PreprocessorConfig(name="squashing_scaler_default",
                   append_original=False,
                   categorical_name="ordinal_very_common_categories_shuffled",
                   global_transformer_name="svd_quarter_components",
                   max_features_per_estimator=500)
# 相比 v2 的 svd，使用更节省内存的 svd_quarter_components
```

**4. 真实数据预训练（Real Data Fine-tuning）**

v2.5 的分类器权重在真实数据集上进行了**进一步微调**（fine-tuned），而回归器仍主要依赖合成数据。这是性能提升的主要来源之一。

可用的 v2.5 模型检查点（`src/tabpfn/model_loading.py`）：

```
tabpfn-v2.5-classifier-v2.5_default.ckpt
tabpfn-v2.5-classifier-v2.5_large-features-L.ckpt    # 大特征量专用
tabpfn-v2.5-classifier-v2.5_large-features-XL.ckpt   # 超大特征量专用
tabpfn-v2.5-classifier-v2.5_large-samples.ckpt       # 大样本量专用
tabpfn-v2.5-classifier-v2.5_real.ckpt                # 真实数据专用
```

**5. Thinking Tokens（实验性）**

新增"思考行"机制（`num_thinking_rows`），在 Transformer 前向传播前插入虚拟"思考"token 以增强推理能力（类似 chain-of-thought）。

### 4.2 v2 与 v2.5 关键差异对比表

| 特性 | TabPFN v2 | TabPFN v2.5 |
|------|-----------|-------------|
| 最大样本 | 10,000 | **50,000** |
| 最大特征 | 500 | **2,000** |
| 预训练数据 | 纯合成 | 合成 + **真实数据微调** |
| 分类器 | 基础权重 | **真实数据微调权重** |
| 回归器 | 基础权重 | 合成数据权重 |
| SVD 变体 | `svd` | **`svd_quarter_components`**（更省内存） |
| 数值变换 | `quantile_uni_coarse` | **`squashing_scaler_default`** |
| 目标变换（回归） | None + safepower | None + safepower |
| 特征处理（> 500） | 不支持 | **每 estimator 子采样 500 特征** |
| 许可证 | Apache 2.0 | **非商业许可（须注意！）** |
| 包版本 | < 6.0.0 | **≥ 6.0.0（现为默认）** |

---

## 五、TabPFN 完整使用指南

### 5.1 安装

```bash
pip install tabpfn
# 或从源码安装
pip install "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git"
```

### 5.2 基础使用——分类

```python
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# 准备数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 使用默认版本（v2.5）
clf = TabPFNClassifier()
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)  # 概率
pred  = clf.predict(X_test)        # 类别标签

# 使用 v2
clf_v2 = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
clf_v2.fit(X_train, y_train)
```

### 5.3 基础使用——回归（适合虚拟量测）

```python
from tabpfn import TabPFNRegressor

reg = TabPFNRegressor()
reg.fit(X_train, y_train)

# 点预测（默认用均值）
y_pred = reg.predict(X_test)

# 分位数预测——获取预测不确定度范围
quantiles = [0.1, 0.5, 0.9]  # 10th 百分位，中位数，90th 百分位
q_preds = reg.predict(X_test, output_type="quantiles", quantiles=quantiles)
# q_preds 是 3 个数组的列表

# 获取完整输出（均值、中位数、众数、分位数）
full_output = reg.predict(X_test, output_type="main")
# full_output["mean"], full_output["median"], full_output["mode"], full_output["quantiles"]
```

**对虚拟量测的特别意义**：`output_type="quantiles"` 可以直接给出预测区间（如 90% 置信区间 = [q10, q90]），这对生产过程中的工艺控制非常有价值。

---

## 六、关键可调参数详解

### 6.1 `n_estimators`（最重要的精度参数）

```python
clf = TabPFNClassifier(n_estimators=8)  # 默认值（v2.1.1 后从 4 改为 8）
```

- **作用**：集成的"prompt"数量。每个 estimator 对输入做不同随机扰动后进行前向传播，最终聚合预测结果。
- **类比**：相当于随机森林的树数量
- **建议**：
  - 快速实验：`n_estimators=4`
  - 正常使用：`n_estimators=8`（默认）
  - 特征数 > 500（v2.5）：`n_estimators=16~32`（确保每个特征被充分覆盖）
  - 极致精度：`n_estimators=32`（推理时间约增加 4 倍）
- **代价**：推理时间和内存线性增长

---

### 6.2 `device`（硬件加速）

```python
clf = TabPFNClassifier(device="auto")       # 自动检测（默认）
clf = TabPFNClassifier(device="cuda")       # 单 GPU
clf = TabPFNClassifier(device="cuda:0")     # 指定第 0 张 GPU
clf = TabPFNClassifier(device=["cuda:0", "cuda:1"])  # 多 GPU 并行
clf = TabPFNClassifier(device="cpu")        # CPU（仅建议样本数 < 1000）
clf = TabPFNClassifier(device="mps")        # Apple Silicon（M1/M2/M3）
```

- **建议**：有 GPU 时一定使用 CUDA。CPU 仅适合极小数据集（< 1000 样本）
- **GPU 内存参考**：8GB VRAM 可处理中等规模数据；16GB+ 可处理 50k 样本

---

### 6.3 `fit_mode`（内存/速度权衡）

```python
clf = TabPFNClassifier(fit_mode="fit_preprocessors")  # 默认
```

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| `"low_memory"` | 每次预测时重新计算预处理 | GPU 内存极少，只需预测一次 |
| `"fit_preprocessors"` | `.fit()` 时缓存预处理结果 | **默认推荐**，多次预测同一训练集 |
| `"fit_with_cache"` | 额外缓存 Transformer 的 KV cache | GPU 内存充足，需要极快的多次预测 |
| `"batched"` | 批量模式（仅用于微调） | 微调场景，不建议手动设置 |

---

### 6.4 `memory_saving_mode`（GPU 内存节省）

```python
clf = TabPFNClassifier(memory_saving_mode="auto")  # 默认，自动判断
clf = TabPFNClassifier(memory_saving_mode=True)    # 强制开启（省内存但稍慢）
clf = TabPFNClassifier(memory_saving_mode=False)   # 强制关闭（快但耗内存）
```

- 当出现 CUDA OOM 错误时，设置为 `True`
- 通过自动分批模型内部计算来降低峰值显存

---

### 6.5 `ignore_pretraining_limits`（突破规模限制）

```python
clf = TabPFNClassifier(ignore_pretraining_limits=False)  # 默认：遵守限制
clf = TabPFNClassifier(ignore_pretraining_limits=True)   # 忽略限制
```

**v2.5 的预训练限制**：

- 最大 50,000 样本
- 最大 2,000 特征
- 最大 10 类别（**此限制不可忽略**）

> **适用场景**：生产数据量在 50k~100k 时，配合 `inference_config={"SUBSAMPLE_SAMPLES": 50000}` 使用。

---

### 6.6 `softmax_temperature`（预测置信度）

```python
clf = TabPFNClassifier(softmax_temperature=0.9)  # 默认
```

- **< 1.0**：预测更"自信"（概率分布更尖锐）
- **= 1.0**：无效果，保持原始 logit 的 softmax
- **> 1.0**：预测更"平滑"（概率更接近均匀）
- **建议**：默认 0.9 对大多数情况表现良好；若需校准概率，使用 `eval_metric` + `tuning_config`

---

### 6.7 `categorical_features_indices`（类别特征指定）

```python
clf = TabPFNClassifier(
    categorical_features_indices=[2, 5, 8]  # 第 2、5、8 列为类别特征
)
```

- **默认**（`None`）：自动推断（根据唯一值数量判断）
- **推断规则**：
  - 唯一值 < 4 → 视为类别特征
  - 唯一值 > 30 → 视为数值特征
  - 中间值根据样本量和启发式规则判断

> **对虚拟量测的建议**：若有已知的工艺参数类别（如设备 ID、配方 ID），建议**明确指定**以提升性能。

---

### 6.8 `eval_metric` + `tuning_config`（指标优化）

```python
from tabpfn import TabPFNClassifier

# 针对 F1 分数优化决策阈值
clf = TabPFNClassifier(
    eval_metric="f1",  # 支持: "accuracy", "f1", "balanced_accuracy", "roc_auc", "log_loss"
    tuning_config={
        "calibrate_temperature": True,     # 校准 softmax 温度
        "tune_decision_thresholds": True,  # 优化决策阈值
    }
)
clf.fit(X_train, y_train)
# fit() 时会自动用交叉验证调优阈值
```

- **适用于**：类别不平衡问题（如良品/不良品检测）
- `balance_probabilities=True`：另一种处理不平衡的简单方式（不需要验证数据）

---

### 6.9 `inference_config`（高级推理配置）

这是最灵活但也最复杂的参数，对应 `InferenceConfig` 类（`src/tabpfn/inference_config.py`）：

```python
from tabpfn import TabPFNClassifier

clf = TabPFNClassifier(
    inference_config={
        # 子采样：大数据集限制每次使用的样本数
        "SUBSAMPLE_SAMPLES": 10000,     # 每个 estimator 最多使用 10000 个样本

        # 特征工程
        "POLYNOMIAL_FEATURES": 10,      # 添加最多 10 个二阶多项式特征
        "FINGERPRINT_FEATURE": True,    # 添加行哈希特征（处理重复行），默认 True

        # 离群值处理
        "OUTLIER_REMOVAL_STD": 12.0,    # 超过 12 倍标准差的视为离群值（分类默认值）

        # 特征/类别扰动策略
        "FEATURE_SHIFT_METHOD": "shuffle",   # "shuffle", "rotate", None
        "CLASS_SHIFT_METHOD": "shuffle",     # 仅分类有效

        # 规模限制（一般不需要修改）
        "MAX_NUMBER_OF_SAMPLES": 50000,
        "MAX_NUMBER_OF_FEATURES": 2000,
    }
)
```

**最常用的 `inference_config` 参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SUBSAMPLE_SAMPLES` | `None` | 大数据集控制内存用量 |
| `POLYNOMIAL_FEATURES` | `"no"` | 添加多项式特征以捕捉非线性关系 |
| `OUTLIER_REMOVAL_STD` | `"auto"` | 分类默认 12.0，回归默认 None |
| `FEATURE_SHIFT_METHOD` | `"shuffle"` | 特征位移方式 |
| `FINGERPRINT_FEATURE` | `True` | 行哈希特征（建议保持 True） |
| `REGRESSION_Y_PREPROCESS_TRANSFORMS` | `(None, "safepower")` | 目标变量预处理 |

---

### 6.10 `random_state`（可复现性）

```python
clf = TabPFNClassifier(random_state=42)   # 固定种子
clf = TabPFNClassifier(random_state=None) # 每次随机（与 sklearn 默认行为相同）
```

- **默认值为 0**（注意：这与 sklearn 的默认 `None` 不同！）
- 即使固定种子，由于 PyTorch 非确定性操作，结果可能有细微差异
- 最佳可复现性配置：`random_state=0` + `inference_precision=torch.float32`

---

### 6.11 `n_preprocessing_jobs`（CPU 并行）

```python
clf = TabPFNClassifier(n_preprocessing_jobs=1)  # 默认，强烈推荐
```

- **建议始终保持默认值 1**
- 仅在 CPU 核心极多且数据量极大时考虑 > 1

---

## 七、对虚拟量测场景的针对性建议

### 7.1 典型应用场景匹配

| VM 场景 | 推荐 TabPFN 用法 |
|---------|-----------------|
| 连续过程参数预测 | `TabPFNRegressor` + `output_type="quantiles"` |
| 良品/不良品分类 | `TabPFNClassifier` + `eval_metric="f1"` + `tuning_config` |
| 少量工程样本（< 500 条） | 直接使用，TabPFN 在小数据集上最优 |
| 多工步数据融合 | 将不同工步特征拼接，指定 `categorical_features_indices` |
| 带置信区间的预测 | `TabPFNRegressor.predict(output_type="quantiles")` |
| 实时在线推断 | `fit_mode="fit_with_cache"` 缓存训练数据 KV cache |
| 半导体产品参数预测 | 考虑微调版本（`FinetunedTabPFNRegressor`） |

### 7.2 推荐配置模板

**通用虚拟量测（回归）配置**：

```python
from tabpfn import TabPFNRegressor

reg = TabPFNRegressor(
    n_estimators=16,              # 较大集成数提升稳定性
    device="cuda",                # GPU 推理
    fit_mode="fit_preprocessors", # 缓存预处理（多次 predict）
    random_state=42,              # 固定随机种子
    ignore_pretraining_limits=True if len(X_train) > 50000 else False,
    inference_config={
        "SUBSAMPLE_SAMPLES": 50000,   # 大数据集时限制子采样
        "OUTLIER_REMOVAL_STD": None,  # 回归不去除离群值（生产数据可能有意义的极端值）
    }
)

reg.fit(X_train, y_train)

# 预测 + 不确定度
y_pred = reg.predict(X_test)
y_intervals = reg.predict(X_test, output_type="quantiles", quantiles=[0.05, 0.95])
print(f"预测值: {y_pred}")
print(f"90% 预测区间: [{y_intervals[0]}, {y_intervals[1]}]")
```

**不平衡分类（如异常检测）配置**：

```python
from tabpfn import TabPFNClassifier

clf = TabPFNClassifier(
    n_estimators=16,
    device="cuda",
    eval_metric="f1",
    tuning_config={"tune_decision_thresholds": True},
    balance_probabilities=True,   # 对不平衡数据有帮助
    random_state=42,
)
clf.fit(X_train, y_train)
```

### 7.3 与其他基线方法的比较参考

根据 TabPFN 论文在 OpenML 基准上的对比结果：

- **vs XGBoost**：小数据集（< 3000 样本）TabPFN 通常更优；大数据集 XGBoost 仍具竞争力
- **vs AutoGluon**：性能相近，但 TabPFN 无需长时间搜索，几秒内出结果
- **vs 深度学习（DNN）**：表格数据上 TabPFN 通常优于普通 DNN，且不需要调超参

### 7.4 已知限制与注意事项

1. **类别上限**：最多支持 **10 个类别**，超过时需要使用 `tabpfn-extensions` 的 `many_class` 模块
2. **v2.5 许可证**：仅限**非商业用途**；商业使用需联系 Prior Labs 获取 Enterprise 授权
3. **GPU 内存**：50k 样本需要约 16GB VRAM；小数据集 8GB 足够
4. **特征数 > 500**：v2.5 下每个 estimator 子采样 500 特征，需增大 `n_estimators`
5. **纯文本特征**：本地版本支持有限，建议先做文本嵌入再输入

---

## 八、快速参考卡片

```python
# 最简单的使用方式（适合快速验证）
from tabpfn import TabPFNClassifier, TabPFNRegressor

# 分类
clf = TabPFNClassifier()
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)

# 回归（含不确定度）
reg = TabPFNRegressor()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
q_preds = reg.predict(X_test, output_type="quantiles", quantiles=[0.05, 0.5, 0.95])

# 关键参数速查
TabPFNClassifier(
    n_estimators=8,               # 集成数量（精度 vs 速度）
    device="auto",                # 硬件
    fit_mode="fit_preprocessors", # 内存模式
    softmax_temperature=0.9,      # 预测置信度
    eval_metric="accuracy",       # 优化指标
    ignore_pretraining_limits=False,  # 是否超规模使用
    random_state=42,              # 可复现性
    inference_config={            # 高级设置
        "SUBSAMPLE_SAMPLES": None,
        "POLYNOMIAL_FEATURES": "no",
    }
)
```

---

## 参考资料

1. Hollmann, N., et al. (2023). *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second*. ICLR 2023.
2. Hollmann, N., et al. (2024). *TabPFN v2: Improved In-Context Learning for Tabular Data*. arXiv:2501.02945.
3. Prior Labs. (2025). *TabPFN 2.5 Model Report*. <https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report>
4. 代码仓库：<https://github.com/PriorLabs/TabPFN>
5. API 文档：<https://priorlabs.ai/docs>
6. Hugging Face 模型页：<https://huggingface.co/Prior-Labs/tabpfn_2_5>

---

*本汇报材料基于 TabPFN 代码库（版本 6.4.1，2026 年 3 月）编写，代码来源于 `src/tabpfn/` 目录下各模块。*
