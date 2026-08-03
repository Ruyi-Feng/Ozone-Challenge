# Baseline 模型训练报告 — 全数据集（inD + CitySim + NGSIM）

**日期**: 2026-08-04
**数据集**: inD (3 scenes) + CitySim (3 scenes) + NGSIM Peachtree
**模型**: BaselineConflictModel（Shared LSTM + 双头预测，memmap 加速）

---

## 1. 技术改动：解决 64 GB CSV 无法加载的问题

### 问题
数据处理产出 `events_data_train.csv` 64 GB，`BaselineConflictDataset.__init__` 用 `pd.read_csv()` 全量加载，在 8 GB 内存机器上直接 OOM 崩溃。

### 方案演进

| 尝试 | 做法 | 结果 |
|------|------|------|
| 1 | CSV → 单个 Parquet，按 Event_id 过滤查询 | 每个 query 64ms，一个 epoch 需 2 小时 |
| 2 | 按 row group 索引加速 | 仍太慢，随机 I/O 严重 |
| **3（采用）** | **预处理为 memmap tensor cache** | 训练集 1087 MB，验证集 310 MB，sub-ms 查询 |

### 最终架构
```
CSV (64G) → Parquet (4.6G) → memmap .npy (1.1G)
                                    ↓
                    BaselineConflictDataset (mmap_mode='r')
                                    ↓
                    __getitem__: 直接数组索引，无需反序列化
```

新增脚本：
- `data_processing/scripts/convert_csv_to_parquet.py` — CSV → Parquet 分块转换
- `data_processing/scripts/build_tensor_cache.py` — Parquet → memmap tensor cache

修改文件：
- `model_baseline/datasets/baseline.py` — 改为 memmap 读取
- `configs/model_baseline.yaml` — 数据路径指向 tensor cache

---

## 2. 数据概况

| 数据集 | 场景数 | 原始行数 | 车辆数 | 平均速度 |
|--------|--------|----------|--------|----------|
| inD | 3 | 82k~143k/scene | 384~694 | 5.3 m/s |
| CitySim | 3 | 54k~166k/scene | 247~481 | 3.7 m/s |
| NGSIM Peachtree | 1 | 873,887 | 1,545 | 4.6 m/s |

### 管线输出

| 指标 | Train | Val |
|------|-------|-----|
| 总样本 | 126,995 | 36,251 |
| Conflict | 119,147 (93.8%) | 34,089 (94.0%) |
| Non-conflict | 7,848 (6.2%) | 2,162 (6.0%) |
| 冲突:非冲突 | **15.2:1** | **15.8:1** |
| 冲突中有 target | 93,286 (78.3%) | 24,944 (73.2%) |

### 各场景冲突比例

| 场景 | 总事件 | 冲突 | 非冲突 | 比例 |
|------|--------|------|--------|------|
| CitySim_A01 | 1,149 | 355 | 794 | **0.4:1** ✅ |
| CitySim_D01 | 275 | 133 | 142 | **0.9:1** ✅ |
| CitySim_E01 | 524 | 185 | 339 | **0.5:1** ✅ |
| inD_00 | 538 | 296 | 242 | **1.2:1** ✅ |
| inD_12 | 1,319 | 810 | 509 | **1.6:1** ✅ |
| inD_26 | 2,568 | 1,573 | 995 | **1.6:1** ✅ |
| **NGSIM Peachtree** | **121,223** | **116,396** | **4,827** | **24.1:1** ❌ |

inD 和 CitySim 的比例均在 0.4~1.6:1，完全正常。**NGSIM Peachtree 占训练集 95%，冲突率 96%，拖垮了整体数据平衡。**

---

## 3. NGSIM Peachtree 数据质量问题

### 3.1 TTC 阈值不适用于拥堵路况

| 指标 | Peachtree | 说明 |
|------|-----------|------|
| 平均速度 | 4.6 m/s (16.6 km/h) | 城市拥堵速度 |
| 速度中位数 | 3.5 m/s (12.6 km/h) | |
| 速度为 0 的比例 | 19.1% | 大量停车等待 |
| 冲突时平均速度 | 3.1 m/s (11.3 km/h) | 低速跟车被误判 |
| 冲突率 | 96.0% | 几乎所有事件都是冲突 |

**根因**：Peachtree 是亚特兰大市中心主干道，交通拥堵严重。在低速跟车场景下，车辆间距常年小于安全距离，TTC < 1.5s 是常态而非安全关键事件。当前 1.5s 的 TTC 阈值适用于高速公路/自由流，不适用于城市拥堵路况。

### 3.2 carId 重复问题

```
受影响: 53,867 个 (carId, frameNum) 对 (6.2% 的原始数据)
表现: 同一 frameNum 下，相同 carId 对应两辆完全不同的车
      carId=2, frameNum=5:
        → 车辆 A: position=(111.4, 556.9), heading=278.3°
        → 车辆 B: position=(188.9, 23.6), heading=98.3°
```

这是 NGSIM 已知的数据质量问题：carId 字段被重复使用。Commit `63e517b` 中已部分修复（precomputation 阶段去重），但根本问题仍存在。

### 3.3 其他问题

- **缺少 scene_id 列**：inD 和 CitySim 有 scene_id，Peachtree 没有，代码用 fallback 处理
- **Heading 范围 [0, 360]**：与 inD/CitySim 的约定不同，但代码中 `_normalize_angle_diff` 已处理

---

## 4. 训练结果

### 训练过程（20 epochs）

```
Epoch   1 | train loss=1.1763 | val loss=0.9298 | c_acc=0.9396 | t_acc=0.5914
Epoch   7 | train loss=0.7760 | val loss=0.8800 | c_acc=0.9475 | t_acc=0.6266
Epoch  11 | train loss=0.6832 | val loss=0.8721 | c_acc=0.9504 | t_acc=0.6403  ← best
Epoch  20 | train loss=0.5788 | val loss=0.9536 | c_acc=0.9442 | t_acc=0.6316
```

**最佳模型**: Epoch 11, val_loss=0.8721

### 最佳结果

| 指标 | 值 |
|------|-----|
| Conflict Accuracy | **95.06%** |
| Conflict Precision | — |
| Conflict Recall | — |
| Conflict F1 | — |
| Target Accuracy | **64.03%** |

### 对比上次（IntersectionD only）

| | 上次 (v2.1) | 这次 (全数据) |
|------|-------------|---------------|
| 数据场景 | IntersectionD ×3 | inD ×3 + CitySim ×3 + Peachtree |
| 样本量 | 314 (val) | 36,251 (val) |
| 帧率 | 25 fps / 200 帧 | 10 fps / 80 帧 |
| 冲突:非冲突 | **~1:1** | **15:1** |
| Conflict Acc | 79.30% | **95.06%** |
| Target Acc | **97.67%** | 64.03% |

Conflict Acc 从 79% 升到 95% 不是模型变好了，而是 **数据不平衡导致模型学会了"全猜 conflict"也能拿 94%**。上次 1:1 平衡时，猜全 0 只有 53%，模型必须真正学习。

Target Acc 从 98% 降到 64% 是因为：
1. Peachtree 数据中 target 角色严重偏斜（right_front 37.9%，rear 3.4%）
2. 模型倾向于预测高频类别，64% ≈ right_front + left_front 的占比

---

## 5. 根因总结

```
全数据 15:1 失衡
    │
    └── NGSIM Peachtree (95% of data, 96% conflict rate)
            │
            ├── TTC < 1.5s 在拥堵路况下不适用
            ├── 低速跟车 (3.1 m/s) 被误判为冲突
            └── carId 重复使用 (6.2% 数据受影响)
```

**inD 和 CitySim 数据质量正常**，冲突比例 0.4~1.6:1，可以直接用于训练。

---

## 6. 建议

1. **短期**：从训练集剔除 Peachtree，仅用 inD + CitySim（约 6k 样本，比例正常）
2. **中期**：为 Peachtree 单独调整冲突检测参数（提高 TTC 阈值或加入速度下限过滤）
3. **长期**：对不同场景类型（高速/城市/拥堵）使用不同的冲突判定标准
