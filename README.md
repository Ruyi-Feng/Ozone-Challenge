# Ozone-Challenge

Data pre-process and baseline model for Ozone Challenge.

## 模块

| 目录 | 说明 |
|------|------|
| `data_processing/` | 从标准化 CSV 提取 conflict / non-conflict 样本 |
| `model_baseline/` | 基于处理后数据的 baseline 模型 |

仓库目录、主线流程与流通数据格式见：[docs/REPO_STRUCTURE.md](docs/REPO_STRUCTURE.md)。  
数据处理操作指南见：[README_使用指南.md](README_使用指南.md)。

---

## Baseline 模型设计

任务：给定未来冲突窗口起点 `t0` 之前的 **8s 多车 history**，预测未来 **3s** 内：

1. 是否发生冲突（`is_conflict`）
2. 冲突对手是哪辆邻车（`target_idx` / slot）

选型：`configs/model_baseline.yaml` 中的 `model.name`（如 `baseline`）同时经 **data factory** 与 **model factory** 选择对应 Dataset 与网络。

### 目录结构

```text
model_baseline/
├── config.py                 # YAML → BaselineRuntimeConfig
├── factories/
│   ├── data_factory.py       # model_name → Dataset
│   └── model_factory.py      # model_name → Model
├── datasets/baseline.py      # 样本契约（实现待填）
├── models/baseline.py        # Shared LSTM + 双头
├── train.py / evaluate.py    # 训练 / 评估（循环待填）
└── scripts/run_{train,eval}.py
```

### 输入 / 输出契约

| 张量 | 形状 | 说明 |
|------|------|------|
| `x` | `[B, A, T, F]` | 多车 history；`A=7`，`F` 建议为相对 ego@t0 的 `(dx, dy, heading_rel, speed)` |
| `agent_mask` | `[B, A]` | slot 是否有车；ego 应为 `True` |
| `conflict_logit` | `[B]` | 冲突二分类 logit（sigmoid 前） |
| `target_logits` | `[B, A-1]` | 六个邻车 slot 的分数；空槽为 `-inf` |

Agent 轴顺序（固定对齐）：

```text
[ego, front, rear, left_front, left_rear, right_front, right_rear]
```

### 网络结构（`baseline`）

```text
x [B, A, T, F]
    │
    ▼  Shared LSTM（逐 agent 独立编码，参数共享）
h [B, A, H]
    │
    ├─ h_ego = h[:, 0]
    └─ h_nbr = h[:, 1:]          # [B, 6, H]
           │
           ├─ masked mean（仅 agent_mask 有效邻车）
           │       → [h_ego ; h_nbr_pool] → MLP → conflict_logit [B]
           │
           └─ 逐邻车 [h_ego ; h_a] → MLP → target_logits [B, 6]
                   （空槽 masked_fill → -inf）
```

- **变长邻车**：每样本邻车数可不同；缺车 slot 零填充 + `agent_mask=False`，池化与 softmax/CE 只看有效槽位。
- **Loss**：`BCE(conflict) + λ · CE(target | is_conflict=1)`（无冲突样本不算 target loss）；见 `BaselineConflictModel.compute_loss`。

### 配置入口

```yaml
# configs/model_baseline.yaml
model:
  name: baseline
  num_agents: 7
  num_features: 4
  hidden_dim: 64
```
