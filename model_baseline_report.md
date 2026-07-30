# Baseline 模型训练报告 — IntersectionD-01

**日期**: 2026-07-30  
**数据集**: CitySim IntersectionD-01（Permissive Left Turn Phasing）  
**模型**: BaselineConflictModel（Shared LSTM + 双头预测）

---

## 1. 数据概况

| 项目 | 值 |
|------|-----|
| 原始数据 | IntersectionD-01.csv（96,433 行） |
| 帧率 | 25 fps |
| 车辆数 | 101 |
| 帧数 | 9,000 |
| OBB 数据 | 有（用于 2D_TTC 碰撞检测） |

### 管线输出

| 指标 | Train | Val |
|------|-------|-----|
| 总样本 | 3,322 | 1,010 |
| Conflict | 3,222 | 970 |
| Non-conflict | 100 | 40 |
| 冲突占比 | 97.0% | 96.0% |

**严重类别失衡**：非冲突样本仅占 3-4%。

---

## 2. 模型结构

```
x [B, 7, 80, 4]          ← 7 辆车、80 帧历史、4 特征 (dx, dy, heading_rel, speed)
      │
      ▼  Shared LSTM (4→64，7 辆车共享权重)
h [B, 7, 64]
      │
      ├─ h_ego = h[:, 0]                           → [B, 64]
      └─ h_nbr = h[:, 1:]                          → [B, 6, 64]
              │
              ├─ masked_mean → h_nbr_pool           → [B, 64]
              │     └─ [h_ego; h_nbr_pool] → MLP    → conflict_logit [B]
              │
              └─ 逐邻车 [h_ego; h_a] → MLP          → target_logits [B, 6]
                                                       (空槽 mask 为 -inf)
```

- 参数量：~83k
- Loss：`BCE(conflict) + CE(target | is_conflict=1)`

---

## 3. 训练结果

| Epoch | Train Loss | Val Loss | Conflict Acc |
|-------|-----------|----------|-------------|
| 1 | 1.2291 | 0.8970 | 96.04% |
| 2 | 0.5609 | 0.6123 | 96.04% |
| 3 | 0.4032 | 0.5679 | 96.04% |
| 4 | 0.3502 | 0.6325 | 96.04% |
| **5** | **0.3211** | **0.4994** ✓ | **96.04%** |
| 6 | 0.2610 | 0.6003 | 97.52% |
| ... | ... | ... | ... |
| 20 | 0.1024 | 0.6759 | 97.52% |

**最优 epoch: 5**（val_loss=0.4994），之后过拟合。

---

## 4. 评估结果（Best Checkpoint, Epoch 5）

```
Samples:    1010
Loss:       0.4994
Accuracy:   0.9604
Precision:  0.9604
Recall:     1.0000
F1:         0.9798
TP=970  TN=0  FP=40  FN=0
```

### 关键发现

**TN=0, FP=40**：模型从未预测"无冲突"。40 个非冲突样本全部被误判为冲突。

**原因分析**：

| 如果把所有样本都预测为"conflict" | 值 |
|------|-----|
| Accuracy | 970/1010 = 96.04% |
| Precision | 970/1010 = 96.04% |
| Recall | 100% |
| F1 | 0.9798 |

模型学到的策略与"闭眼全猜 conflict"的 naive baseline **完全一致**。BCE loss 在 97% 冲突占比下，只需将 logit 推到略大于 0 即可获得低 loss。

**Target head 有在学**：target_loss 从 1.06 → 0.10，说明冲突对手识别在进步，但 eval 未单独评估此指标。

---

## 5. 根因

`data_processing/core/conflict_detect.py` 第 701 行：

```python
max_per_ego = 20  # cap to avoid explosion
```

- **Conflict**：TTC 物理检测，穷举无上限
- **Non-conflict**：随机采样，每车最多 20 个 → 101 车最多 2,020 个

冲突检出 35k+，非冲突最多 2k，比例严重失衡。

---

## 6. 后续改进方向

1. 提高 `max_per_ego` 上限（20 → 200），增加非冲突采样
2. Loss 添加类别权重，惩罚非冲突误判
3. Eval 增加 target 头准确率评估
4. 合并多个 Intersection 场景丰富数据多样性
