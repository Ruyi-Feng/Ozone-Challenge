# 跨场景交叉污染 Bug 修复报告

**日期**: 2026-07-31
**影响范围**: `data_processing/core/conflict_detect.py` — `_find_front_pairs()`、`detect_conflicts()`、`sample_non_conflicts()`

---

## 1. Bug 描述

### 1.1 触发条件

多文件合并处理时（`run_pipeline` 将 IntersectionD-01/02/03 三个 CSV 各标记 `scene_id` 后 `pd.concat`），三个场景的原始数据共享同一套 `frameNum`（0–8999），但车辆和位置完全不同：

| 文件 | carId 范围 | 车辆数 | 帧范围 |
|------|-----------|--------|--------|
| IntersectionD-01.csv | 0 – 213 | 101 | 0 – 8999 |
| IntersectionD-02.csv | 187 – 440 | 113 | 0 – 8999 |
| IntersectionD-03.csv | 404 – 635 | 101 | 0 – 8999 |

合并后 `frameNum=1272` 的一帧中会同时包含 D-01 和 D-02 的车辆。冲突检测的逐帧配对 (`_find_front_pairs`) **没有按 `scene_id` 隔离**，导致 D-01 的 ego 与 D-02 的车辆被错误配对，计算出虚假的 TTC 冲突。

### 1.2 具体案例

```
frameNum=1272:
  IntersectionD-01: ego=5, car_0, car_6, car_13, ...
  IntersectionD-02: car_204, car_187, car_210, ...
                         ↓
  _find_front_pairs 将 (ego=5, target=204) 视为同一场景的前方配对
                         ↓
  TTC 计算 → TTC < 1.5s → 标记为 conflict
                         ↓
  conflict_target_id = 204（但 204 是 D-02 的车，在 D-01 中不存在于该帧）
```

### 1.3 传播链

```
Stage 1 (冲突检测):
  _find_front_pairs     ← 无 scene_id 隔离 → 跨场景交叉配对
  detect_conflicts      ← 同上
  ↓
  conflict_target_id = 跨场景车辆（虚假冲突）

Stage 2 (窗口验证):
  不受影响（只验证 ego 轨迹连续性，不涉及 target）

Stage 3 (邻车提取):
  extract_neighbors_for_event  ← 有 scene_id 隔离（_frame_rows 按 scene_id 过滤）
  ↓
  在 target 所属场景中找不到该车辆 → target 不在 event data 中

Stage 4 (数据集构建):
  BaselineConflictDataset.__getitem__
  ↓
  slot_cars 中没有 conflict_target_id → target_idx = -1
  ↓
  target 头的 CE loss 对这些样本不参与训练
```

---

## 2. 修复前后对比

### 2.1 Target 覆盖率

| 指标 | 修复前 | 修复后 | 改善 |
|------|--------|--------|------|
| Val 冲突样本数 | 256 | 118 | — |
| Target 在 event data 中 | 55 (21.5%) | 113 (95.8%) | **+74.3 pp** |
| Target 不在 event data（跨场景虚假） | 196 (76.6%) | 0 (0%) | **消除** |
| Target 不在 event data（历史窗口无轨迹，合理） | — | 5 (4.2%) | — |
| 无效 target_id（不在 raw data 中） | 5 (2.0%) | 0 (0%) | **消除** |

> **注**: 修复后剩余 5 个 missing target 属于合理边界情况——冲突对手在 `t_conflict` 时刻或之后才首次在场景中出现，历史窗口 `[t0 - 8s, t0]` 内无该车辆的任何轨迹，因此 `neighbor_filter` 无法将其纳入邻车位。这是数据本身的特性，非 bug。

### 2.2 数据分布

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| **Train** | | |
| 总事件数 | 1,933 | ← 持平 |
| 冲突事件 | — | 752 (38.9%) |
| 非冲突事件 | — | 1,181 (61.1%) |
| **Val** | | |
| 总事件数 | 434 | 358 |
| 冲突事件 | 256 (59.0%) | 118 (33.0%) |
| 非冲突事件 | 178 (41.0%) | 240 (67.0%) |

修复前冲突占比 59% 是因为跨场景虚假配对大量产生假冲突（196/256 = 76.6% 为虚假）。修复后冲突占比降至 33%（Val），更符合真实交通场景的冲突发生率。

### 2.3 Val Target Role 分布（修复后）

| Role | 数量 | 占比 |
|------|------|------|
| left_front | 55 | 48.7% |
| right_front | 39 | 34.5% |
| front | 18 | 15.9% |
| left_rear | 1 | 0.9% |
| rear | 0 | 0% |
| right_rear | 0 | 0% |

冲突对手集中在前方（front/left_front/right_front 合计 99.1%），后方冲突极少——这与 TTC 检测只扫描前方车辆（`longitudinal ≥ 0`）的设计一致。

---

## 3. 修复方案

### 3.1 修改的文件

**`data_processing/core/conflict_detect.py`** — 三处修改：

| 函数 | 修改 |
|------|------|
| `_find_front_pairs()` | 新增 scene_id 分组：若 frame_df 含 `scene_id` 列，先 `groupby("scene_id")` 再逐场景调用向量化配对 |
| `detect_conflicts()` | 新增 scene_id 循环：若含 `scene_id` 列，逐场景调用 `_detect_conflicts_one_scene()`，场景间清空速度/加速度缓存 |
| `sample_non_conflicts()` | 同上：逐场景调用 `_sample_non_conflicts_one_scene()`，exclude 列表按 `scene_id` 过滤 |

### 3.2 `_find_front_pairs` 修复

**修复前**（第 465 行）：
```python
def _find_front_pairs(frame_df, cfg):
    # 直接对 frame_df 全体做 numpy 向量化配对
    # 不区分 scene_id → 跨场景车辆被错误配对
    x = frame_df["carCenterXm"].to_numpy(...)
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    ...
```

**修复后**：
```python
def _find_front_pairs(frame_df, cfg):
    if "scene_id" in frame_df.columns:
        pairs = []
        for _, scene_df in frame_df.groupby("scene_id"):
            pairs.extend(_find_front_pairs_single(scene_df, cfg))
        return pairs
    return _find_front_pairs_single(frame_df, cfg)
```

向量化计算逻辑完整保留在 `_find_front_pairs_single()` 中，零性能损失。`groupby("scene_id")` 的 Python 开销可忽略（每帧最多 3 个 scene）。

### 3.3 `detect_conflicts` 修复

**修复前**：单次调用处理全量合并数据，`scene_id` 只取 `unique()[0]` 用于 `ConflictCandidate` 填充，所有 pair 共享同一个 scene_id。

**修复后**：
```python
def detect_conflicts(raw_df, cfg):
    if "scene_id" in raw_df.columns:
        all_candidates = []
        for scene_id in sorted(raw_df["scene_id"].unique()):
            _clear_caches()  # ← 场景切换时清空速度/加速度缓存
            scene_df = raw_df[raw_df["scene_id"] == scene_id]
            candidates = _detect_conflicts_one_scene(scene_df, cfg, str(scene_id))
            all_candidates.extend(candidates)
        return all_candidates
    return _detect_conflicts_one_scene(raw_df, cfg, "scene")
```

关键点：
- `_clear_caches()` 在场景切换时调用，防止前一个场景的速度/加速度缓存污染下一个场景
- 核心检测逻辑提取到 `_detect_conflicts_one_scene()`，零行为变化
- pair_series 的 key `(ego_id, tgt_id)` 在单场景内天然唯一

### 3.4 `sample_non_conflicts` 修复

同样模式：逐场景采样，`exclude` 列表按 `scene_id` 过滤，防止一个场景的冲突窗口阻塞另一个场景同 carId 的采样。

### 3.5 为什么只改 conflict_detect.py

问题的根源只在 Stage 1。Stage 2（`trajectory_window.py`）处理 ego 级别的窗口验证，不涉及车辆配对。Stage 3（`neighbor_filter.py`）已经有正确的 `scene_id` 隔离——`_frame_rows()` 和 `_precompute_frame_neighbors()` 都按 `(scene_id, frameNum)` 做 key。Stage 4（export）直接从 Stage 3 的输出构建表，不受影响。

**修复策略**：在错误发生的位置（Stage 1）阻断，而不是在下游做弥补。

---

## 4. 性能影响

| 项目 | 说明 |
|------|------|
| 每帧 groupby 开销 | 3 个 scene × 9000 帧 = 27000 次 `groupby`，每次 < 0.1ms |
| 场景级循环开销 | `detect_conflicts` 内层循环 3 次，各处理约 3000 帧 |
| 向量化配对 | 每场景车辆数约为合并数据的 1/3，`O(N²)` 总量下降（3 × (N/3)² = N²/3 vs N²） |
| 总体耗时 | 相比修复前无明显变化（瓶颈仍在 OBB 几何计算） |

---

## 5. 后续需要

1. **重新训练模型**: 旧 checkpoint（epoch 6, val_loss=0.7114）在污染数据上训练，76.6% 的冲突样本 target 是错误的跨场景配对。需要在新数据上重新训练。
2. **Target 头评估**: 修复后 target 覆盖率达 95.8%，可以充分利用 target 头。预期 target accuracy 相比之前的 47-53% 有显著改善。
3. **剩余 4.2% 的处理**: 5 个 target 不在 event data 的样本是因为冲突对手在历史窗口无轨迹。可考虑：
   - 在数据集中对这些样本设置 `target_idx = -1`，跳过 target loss（当前已实现）
   - 或扩大 neighbor_filter 的时间窗口，把未来窗口中的车辆也纳入邻车 slot
