# Ozone-Challenge 项目完整方法论与文件职责

**最后更新**: 2026-07-20

---

## 一、项目目标

这是一个 **轨迹预测 + 冲突检测** 的数据处理与模型训练流水线。输入是标准化车辆轨迹 CSV（NBDT 格式），输出是可用于训练的样本集（data + label CSV），最终训练一个模型，在给定前 8 秒历史轨迹的前提下，预测未来 3 秒内是否会发生冲突。

```
原始轨迹 CSV  →  [数据处理流水线]  →  data/label CSV  →  [模型训练]
```

---

## 二、全局目录结构

```
Ozone-Challenge/
├── configs/
│   ├── data_processing.yaml      # 数据处理参数（窗口时长、距离阈值、TTC阈值等）
│   └── model_baseline.yaml       # 模型超参（TODO）
│
├── data/                         # 运行时数据（gitignore）
│   ├── raw/                      # 输入：NBDT 标准化轨迹 CSV
│   ├── interim/candidates/       # 中间产物：conflict/non-conflict 候选列表
│   └── processed/
│       ├── data/events_data.csv  # 输出：history 轨迹特征（模型输入）
│       └── labels/
│           ├── events_labels.csv       # 输出：每事件一行 label
│           └── events_future_traj.csv  # 输出：未来 3s 轨迹
│
├── data_processing/              # ══ 模块一：数据处理 ══
│   ├── pipeline.py               # 主线编排
│   ├── core/                     # 核心算法
│   │   ├── conflict_detect.py    # Stage 1: 冲突检测
│   │   ├── trajectory_window.py  # Stage 2: 时间窗口校验
│   │   ├── neighbor_filter.py    # Stage 3: 周边车筛选
│   │   └── export.py             # Stage 4: 导出 CSV
│   ├── io/                       # I/O 与格式约定
│   │   ├── schema.py             # 列名契约 + 中间数据结构
│   │   ├── readers.py            # 读 raw CSV / YAML
│   │   └── writers.py            # 写 processed CSV
│   ├── utils/                    # 通用工具
│   │   ├── geometry.py           # 距离、相对位姿、邻车位分类
│   │   └── time_utils.py         # 秒↔帧转换、连续覆盖检查
│   └── scripts/                  # CLI 入口
│       ├── run_pipeline.py       # 全流程
│       ├── run_conflict_detect.py
│       ├── run_window_check.py
│       └── run_neighbor_extract.py
│
├── model_baseline/               # ══ 模块二：模型 ══
│   ├── dataset.py                # TODO
│   ├── train.py                  # TODO
│   ├── evaluate.py               # TODO
│   └── models/baseline.py        # TODO
│
├── tests/data_processing/        # 单元测试
└── docs/REPO_STRUCTURE.md        # 设计文档
```

---

## 三、数据处理流水线 — 四阶段

### 时间窗口定义

```
                    t_conflict
                        │
    ═════════════════════╪══════════════════
    ←── 8s history ──→ t0 ←── 3s future ──→
    [t0-8s, t0]         (t0, t0+3s]
```

- **t0**：预测起点。history 窗口结束时刻，future 窗口开始时刻
- **history 8s**：模型可见的输入轨迹
- **future 3s**：模型需要预测的区间，冲突必须落在此区间内

---

### Stage 1 — `conflict_detect.py`：冲突检测

**输入**：`pd.DataFrame`（标准化 NBDT 轨迹，按 frameNum 排序）

**输出**：`List[ConflictCandidate]`（含冲突和非冲突候选）

**方法论**：基于 NBDT `ssm.py` 参考实现的 2D_TTC（二维碰撞时间）

```
detect_conflicts(raw_df, cfg):
  for each frame:
    1. _find_front_pairs() → 找出所有 (ego, target) 对
       - 对每对 (ego, target)：target 需在 ego 前方六方位 slot 内
       - ego→target 距离 ≤ max_distance_m (200m)
    2. for each pair:
       _compute_2d_ttc(ego_row, target_row, dt):
         a. _calculate_nearest_points() → OBB 最近点 + 距离
            - SAT 碰撞检测判定两 OBB 是否相交
            - 分离时：32 个 vertex→edge 候选取最小距离
         b. _rear_forward_strips_intersect() → 几何前置条件
            - 取两车 rear edge，做 4 类检查：
              ① point_in_forward_strip（rear 顶点在对方前向带内）
              ② ray_intersect_segment（前向射线与对方 rear edge 求交）
              ③ two_rays_intersect_forward（双前向射线相交）
              ④ line_segment_intersection（rear edge 直接相交）
         c. _compute_2d_ttc_kernel() → 运动学求解
            - 将速度+加速度投影到 closing direction
            - 加角速度修正：closing = v_rel + (ω₁+ω₂)·dist
            - 解二次方程：dist = closing·t + ½·a_rel·t²
            - 取最小正根 → 2D_TTC
    3. 当 2D_TTC < conflict_ttc_threshold (3s) → 冲突帧
    4. _group_into_events() → 连续冲突帧聚合为事件

  sample_non_conflicts(raw_df, cfg):
    - 对每个 ego，在无冲突的安全区间随机采样 t0
    - 采样数 ≈ 冲突数，每 ego 上限 20 个
```

#### 关键函数清单

| 函数 | 来源 | 职责 |
|------|------|------|
| `_order_rect_points` | NBDT `GeometryHelper` | 4 角点 → CCW 排序 |
| `_rectangles_intersect` | NBDT `GeometryHelper` | SAT 碰撞检测 |
| `_line_segment_intersection` | NBDT `GeometryHelper` | 线段交点 |
| `_is_inside_rect` | NBDT `GeometryHelper` | 点在凸多边形内 |
| `_calculate_nearest_points` | NBDT `GeometryHelper` | OBB 最近点 + 距离 |
| `_rear_edge_from_ordered_bbox` | NBDT `InstantSSMCalculator` | 根据 heading 投影找 rear edge |
| `_point_in_forward_strip` | NBDT `InstantSSMCalculator` | rear 顶点是否在对方前向带内 |
| `_ray_intersect_segment` | NBDT `InstantSSMCalculator` | 前向射线 vs rear edge 线段 |
| `_two_rays_intersect_forward` | NBDT `InstantSSMCalculator` | 双前向射线相交 |
| `_rear_forward_strips_intersect` | NBDT `InstantSSMCalculator` | 4 种检查的 strip 交叉入口 |
| `_project_to_line` | NBDT `InstantSSMCalculator` | 速度/加速度投影到 closing direction |
| `_compute_2d_ttc_kernel` | NBDT `InstantSSMCalculator` | 二次方程 TTC 求解 |
| `_compute_2d_ttc` | 本项目入口 | 串联几何+运动学求单帧 TTC |
| `_find_front_pairs` | 本项目 | 按六方位 slot 构建 pair |
| `_group_into_events` | 本项目 | 连续冲突帧→事件 |
| `detect_conflicts` | 本项目 API | 全量冲突检测 |
| `sample_non_conflicts` | 本项目 API | 非冲突候选采样 |

#### 加速缓存

| 缓存 | key | value | 用途 |
|------|-----|-------|------|
| `_VELOCITY_CACHE` | carId | (prev_heading_deg, dt) | 角速度 deg/s |
| `_ACCEL_CACHE` | carId | (prev_speed_m_s, dt) | 标量加速度 m/s² |

两个缓存均按 frame-by-frame 顺序更新，每帧处理前需 `_clear_caches()`。

---

### Stage 2 — `trajectory_window.py`：时间窗口校验

**输入**：`ConflictCandidate[]` + `raw_df`

**输出**：`List[WindowedEventCandidate]`（通过 8s+3s 校验的候选）

**方法论**：

```
build_windowed_events(raw_df, candidates, cfg):
  for each candidate:
    1. propose_t0:
       - 冲突样本：随机选 t0 使 t_conflict ∈ (t0, t0+3s]
       - 非冲突样本：直接用 meta 中的 sample_frame 作为 t0

    2. validate_history_length:
       - 检查 ego 在 [t0-8s, t0] 内有连续轨迹覆盖
       - 最大允许 gap：2.1 帧间隔

    3. validate_future_interval:
       - 冲突样本：确认 t_conflict 确实在 (t0, t0+3s] 内
       - 非冲突样本：确认 ego 在 future 窗口内有轨迹

    若 history_ok AND future_ok → 保留；否则丢弃
```

---

### Stage 3 — `neighbor_filter.py`：周边车筛选与轨迹采集

**输入**：`WindowedEventCandidate[]` + `raw_df`

**输出**：`List[TrackedNeighborhoodEvent]`（含 ego + 邻车完整轨迹）

**方法论**：

```
extract_neighbors_for_event(raw_df, window, cfg):
  1. find_entered_neighbor_ids:
     - 扫描 [t0-8s, t0] 每一帧
     - 对每帧内所有非 ego 车辆：
       - 距离 ≤ 200m → 计算相对位姿
       - 落入六方位任一 slot → 标记为 neighbor

  2. assign_neighbor_roles:
     - 对每个 neighbor，统计在各 slot 出现的频次
     - 取出现最多的 slot 作为其 role
     - tie-break：取距离最近的帧对应的 slot

  3. collect_persistent_tracks:
     - 一旦标记为 neighbor，在整个 [t0-8s, t0+3s] 区间全程采集其轨迹
     - （不是只采"在 slot 内"的帧）

  4. 确认 conflict_target_role（冲突目标在哪个 slot）
```

#### 六方位 slot

| slot | 含义 | 判断条件 |
|------|------|---------|
| `front` | 正前方 | longitudinal ≥ 0, \|lateral\| ≤ 2m |
| `rear` | 正后方 | longitudinal < 0, \|lateral\| ≤ 2m |
| `left_front` | 左前方 | longitudinal ≥ 0, lateral > 2m |
| `left_rear` | 左后方 | longitudinal < 0, lateral > 2m |
| `right_front` | 右前方 | longitudinal ≥ 0, lateral < -2m |
| `right_rear` | 右后方 | longitudinal < 0, lateral < -2m |

---

### Stage 4 — `export.py`：导出 CSV

**输入**：`TrackedNeighborhoodEvent[]`

**输出**：三个 CSV 文件

| 输出文件 | 粒度 | 关键列 |
|---------|------|--------|
| `events_data.csv` | 每行 = 某 Event 下某车某一帧 | Event_id, scene_id, frameNum, carId, role, t_rel, carCenterXm, carCenterYm, heading, speed |
| `events_labels.csv` | 每行 = 一个 Event | Event_id, scene_id, t0, is_conflict, conflict_target_id, conflict_target_role, t_conflict |
| `events_future_traj.csv` | 每行 = 某 Event 下某车 future 3s 内一帧 | Event_id, carId, frameNum, t_rel, carCenterXm, carCenterYm, heading, speed |

**关联关系**：三个表通过 `Event_id` 关联。`events_data.csv` 只含 history（t_rel ≤ 0），`events_future_traj.csv` 只含 future（t_rel > 0）。

---

## 四、各文件详细职责

### `io/schema.py` — 数据契约（唯一真相源）

| 内容 | 说明 |
|------|------|
| `RAW_REQUIRED_COLUMNS` | 原始 CSV 必须包含的列（frameNum, carId, carCenterXm, carCenterYm, heading, speed, objClass） |
| `RAW_OBB_COLUMNS` | OBB 可选列（boundingBox1Xm..4Ym） |
| `DATA_COLUMNS` | events_data.csv 列定义 |
| `LABEL_COLUMNS` | events_labels.csv 列定义 |
| `FUTURE_TRAJ_COLUMNS` | events_future_traj.csv 列定义 |
| `ConflictCandidate` | Stage 1 输出结构（scene_id, ego_id, is_conflict, t_conflict, conflict_target_id, meta） |
| `WindowedEventCandidate` | Stage 2 输出结构（加 t0, t_end, history_ok, future_ok） |
| `TrackedNeighborhoodEvent` | Stage 3 输出结构（加 history_tracks, future_tracks, neighbor_roles, conflict_target_role） |
| `ProcessingConfig` | 运行时配置 dataclass（history_sec, future_sec, fps, conflict_ttc_threshold, max_distance_m, neighbor_slots, 各输出路径） |
| `NEIGHBOR_SLOTS` | 六方位 slot 元组 |
| `validate_raw_schema(df)` | 校验 raw CSV 列完整性 |

### `io/readers.py` — 数据读取

| 函数 | 用途 |
|------|------|
| `load_config(yaml_path)` → `ProcessingConfig` | 从 YAML 加载配置，处理默认值 |
| `load_raw_csv(path)` → `DataFrame` | 读原始 CSV + 自动 schema 校验 |
| `list_raw_csv_files(dir)` → `Path[]` | 列目录下所有 CSV |

### `io/writers.py` — 数据写出

| 函数 | 用途 |
|------|------|
| `write_events_data(df, path)` | 写 history 轨迹表（自动创建父目录） |
| `write_events_labels(df, path)` | 写 per-event label 表 |
| `write_future_traj(df, path)` | 写 future 3s 轨迹表 |
| `write_interim_candidates(df, path)` | 写中间候选列表（调试用） |

### `utils/geometry.py` — 几何工具

| 函数 | 签名 | 用途 |
|------|------|------|
| `compute_distance` | (x1, y1, x2, y2) → float | 欧几里得距离 |
| `compute_relative_pose` | (ego_x, ego_y, ego_heading, other_x, other_y) → (longitudinal, lateral) | 目标车坐标变换到 ego 坐标系 |
| `classify_neighbor_slot` | (dx, dy, lateral_threshold, slots) → str\|None | 根据 (longitudinal, lateral) 判定六方位 slot |
| `is_within_max_distance` | (distance_m, max_distance_m) → bool | 距离阈值判断 |

**heading 惯例**：`compute_relative_pose` 使用 heading 0°=北（+y）、顺时针增加的惯例，旋转公式为：
- `longitudinal = dx·sin(θ) + dy·cos(θ)`（heading 方向的投影）
- `lateral = -dx·cos(θ) + dy·sin(θ)`（垂直 heading 方向的投影）

### `utils/time_utils.py` — 时间工具

| 函数 | 用途 |
|------|------|
| `sec_to_frames(seconds, fps)` → int | 秒→帧数 |
| `frames_to_sec(frames, fps)` → float | 帧数→秒 |
| `get_track_time_span(track_df, time_col)` → (t_min, t_max) | 单 track 的时间范围 |
| `has_continuous_coverage(timestamps, t_start, t_end, max_gap_sec)` → bool | 检查时间戳序列是否连续覆盖区间 |
| `to_relative_time(timestamps, t0)` → Series | 绝对时间→相对 t0 时间 |
| `window_bounds(t0, history_sec, future_sec)` → (start, t0, end) | 计算 history/future 窗口边界 |

### `pipeline.py` — 主线编排

```
run_pipeline(cfg, raw_path):
  raw_df = load_raw_csv()
  candidates = detect_all_candidates(raw_df, cfg)           # Stage 1
  windows = build_windowed_events(raw_df, candidates, cfg)  # Stage 2
  tracked = extract_neighbors_for_events(raw_df, windows, cfg)  # Stage 3
  return export_events(tracked, cfg)                        # Stage 4
```

子函数：`process_raw_file`（单文件）、`process_raw_dataframe`（DataFrame 入参）、`load_pipeline_config`（别名）。

### 脚本入口

| 脚本 | 用法 |
|------|------|
| `run_pipeline.py` | `python -m data_processing.scripts.run_pipeline --config configs/data_processing.yaml --raw data/raw/scene.csv` |
| `run_conflict_detect.py` | 只跑 Stage 1，输出 candidates CSV |
| `run_window_check.py` | 跑 Stage 1→2 |
| `run_neighbor_extract.py` | 跑 Stage 1→2→3 |

---

## 五、模型模块（TODO）

```
model_baseline/
├── dataset.py       # 读 events_data.csv + events_labels.csv，按 Event_id join → PyTorch Dataset
├── models/baseline.py  # 网络结构（LSTM/Transformer 编码 multi-agent history，输出 is_conflict logit）
├── train.py         # 训练循环
└── evaluate.py      # 评估指标（accuracy, precision, recall, F1）
```

模型任务本质上是**二分类**：给定 8s 多车轨迹 history（ego + 周边车在各 slot，每条轨迹含 frameNum, carId, role, x, y, heading, speed），预测未来 3s 内是否发生冲突。输入是多车时间序列，输出是 `is_conflict ∈ {0, 1}`。

---

## 六、当前完成状态

| 模块 | 文件 | 状态 |
|------|------|------|
| Config | `configs/data_processing.yaml` | ✅ 已配置 |
| Config | `configs/model_baseline.yaml` | ❌ TODO |
| Schema | `io/schema.py` | ✅ 已实现 |
| Readers | `io/readers.py` | ✅ 已实现 |
| Writers | `io/writers.py` | ✅ 已实现 |
| Geometry utils | `utils/geometry.py` | ✅ 已实现 |
| Time utils | `utils/time_utils.py` | ✅ 已实现 |
| **Stage 1** | `core/conflict_detect.py` | ✅ NBDT 标准实现，18/18 测试通过 |
| **Stage 2** | `core/trajectory_window.py` | ✅ 已实现 |
| **Stage 3** | `core/neighbor_filter.py` | ✅ 已实现 |
| **Stage 4** | `core/export.py` | ✅ 已实现 |
| Pipeline | `pipeline.py` | ✅ 已实现 |
| Scripts | `scripts/*` | ✅ 已实现 |
| Model | `model_baseline/*` | ❌ TODO |
| Real data | `data/raw/` | ❌ 待放入真实数据 |
| Tests | `tests/data_processing/test_conflict_detect_fix.py` | ✅ 18/18 通过 |

---

## 七、与 NBDT 参考实现的关系

| 维度 | 参考项目（inD/CitySim/NGSIM 分析） | 本项目（Ozone-Challenge） |
|------|-----------------------------------|--------------------------|
| **目标** | 数据集分析：统计 SSM 指标、对比三数据集 | 训练数据生成：产出可直接训练 model 的 data+label CSV |
| **冲突检测** | NBDT `ssm.py` → 全量 pair 的瞬时 SSM（TTC, 2D_TTC, PET, DRAC） | 复用 NBDT 2D_TTC 算法核心，但只做 conflict/non-conflict 二元判定 |
| **冲突类型** | `_classify_conflict_type`（OBB 最近点+heading 判 head_on/rear_end/angled/side_swipe） | 不需要（label 只需 is_conflict: 0/1） |
| **冲突地点** | 轨迹交汇法（前向射线 ray-casting）判 interior/approach | 不需要 |
| **TET/CPI/DRAC** | 完整计算 | 不需要 |
| **周边车** | 无 | 六方位 slot + 200m + persistent track 采集 |
| **时间窗口** | 无 | 8s history + 3s future 窗口裁剪 |
| **产出** | 统计报告（Markdown） | data/label/future_traj 三个 CSV 文件 |

### 2D_TTC 算法对齐情况

本项目的 `conflict_detect.py` 中以下函数直接对齐 NBDT `ssm.py` 的实现：

| 本项目函数 | NBDT 对应 | 对齐程度 |
|-----------|----------|---------|
| `_order_rect_points` | `GeometryHelper.order_rect_points` | ✅ 逐行对齐 |
| `_line_segment_intersection` | `GeometryHelper.line_segment_intersection` | ✅ 逐行对齐 |
| `_is_inside_rect` | `GeometryHelper._is_inside_rect` | ✅ 逐行对齐 |
| `_rectangles_intersect` | `GeometryHelper._rectangles_intersect` | ✅ 逐行对齐 |
| `_calculate_nearest_points` | `GeometryHelper.calculate_nearest_points` | ✅ 逐行对齐 |
| `_rear_edge_from_ordered_bbox` | `InstantSSMCalculator._rear_edge_from_ordered_bbox` | ✅ 逐行对齐 |
| `_point_in_forward_strip` | `InstantSSMCalculator._point_in_forward_strip` | ✅ 逐行对齐 |
| `_ray_intersect_segment` | `InstantSSMCalculator._ray_intersect_segment` | ✅ 逐行对齐 |
| `_two_rays_intersect_forward` | `InstantSSMCalculator._two_rays_intersect_forward` | ✅ 逐行对齐 |
| `_rear_forward_strips_intersect` | `InstantSSMCalculator._rear_forward_strips_intersect` | ✅ 逐行对齐 |
| `_project_to_line` | `InstantSSMCalculator._project_to_line` | ✅ 逐行对齐 |
| `_compute_2d_ttc_kernel` | `InstantSSMCalculator._compute_2d_ttc_kernel` | ✅ 逐行对齐 |

### 本项目新增（NBDT 中无对应）

| 函数 | 说明 |
|------|------|
| `_compute_2d_ttc(ego_row, target_row, dt)` | 串联 OBB 提取 + 最近点 + strip 检测 + TTC 内核的入口 |
| `_find_front_pairs(frame_df, cfg)` | 逐帧构建 ego→前方 target 的 pair 列表 |
| `_group_into_events(conflict_frames)` | 连续冲突帧聚合为事件 |
| `_scalar_acceleration(car_id, speed, dt)` | 从连续速度帧差分计算标量加速度 |
| `_angular_velocity(car_id, heading_deg, dt)` | 从连续 heading 帧差分计算角速度 (deg/s) |
| `detect_conflicts(raw_df, cfg)` | 全量冲突检测：遍历帧→pair→TTC→事件 |
| `sample_non_conflicts(raw_df, cfg)` | 非冲突候选采样 |
| `detect_all_candidates(raw_df, cfg)` | 合并冲突+非冲突候选 |

---

## 八、测试覆盖

`tests/data_processing/test_conflict_detect_fix.py` — 18 个测试用例：

| 测试类 | 测试 | 覆盖内容 |
|--------|------|---------|
| `TestGeometryHelpers` | 5 个 | CCW 排序、SAT 碰撞/分离、最近点（分离+相交） |
| `TestStripIntersection` | 3 个 | 平行同向、对向、垂直 heading 的 strip 交叉 |
| `Test2DTTCKernel` | 4 个 | 等速 closing、分离返回 None、加速度缩短 TTC、角速度修正 |
| `TestCompute2DTTC` | 3 个 | 完整 TTC 入口（平行 closing、分离、多帧缓存） |
| `TestDetectConflicts` | 3 个 | 端到端：closing 有冲突、分离无冲突、TTC>阈值无冲突 |
