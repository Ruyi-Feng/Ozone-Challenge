# Ozone-Challenge 仓库结构设计

本文档描述仓库的目录组织、数据处理主线流程，以及流通数据格式约定。  
代码与字段细节暂不实现，仅作为后续开发的结构蓝图。

---

## 1. 总体目标

仓库分为两大模块：

| 模块 | 目录 | 职责 |
|------|------|------|
| 数据处理 | `data_processing/` | 从标准化 CSV 提取 conflict / non-conflict sample，产出训练可用的 data + label |
| 模型 Baseline | `model_baseline/` | 基于处理后的数据训练与评估预测模型 |

数据与标签建议**分文件存放**，通过 `Event_id` 关联。

---

## 2. 推荐目录结构

```text
Ozone-Challenge/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt                 # 或 pyproject.toml
├── configs/                         # 全局 / 分模块配置
│   ├── data_processing.yaml         # 时长、距离阈值、邻车位姿定义等
│   └── model_baseline.yaml
│
├── docs/
│   └── REPO_STRUCTURE.md            # 本文件
│
├── data/                            # 运行时数据根目录（大文件建议 gitignore）
│   ├── raw/                         # 原始标准化 CSV（输入）
│   │   └── <scene_or_file>.csv
│   ├── interim/                     # 中间产物（可选：候选事件、筛选日志）
│   │   └── candidates/
│   └── processed/                   # 最终样本
│       ├── data/                    # 轨迹特征（输入侧）
│       │   └── events_data.csv
│       └── labels/                  # 标签（输出侧）
│           └── events_labels.csv
│
├── data_processing/                 # ========== 模块一：数据处理 ==========
│   ├── __init__.py
│   ├── pipeline.py                  # 主线编排：串联各阶段（调用逻辑，不堆实现）
│   ├── core/                        # 核心处理逻辑（可被 pipeline / CLI 复用）
│   │   ├── __init__.py
│   │   ├── conflict_detect.py       # conflict / non-conflict 检测
│   │   ├── trajectory_window.py     # 8s history + 3s future 窗口校验与裁剪
│   │   ├── neighbor_filter.py       # 周边车筛选（位姿 + 200m）与持续采集
│   │   └── export.py                # 写出 data / label CSV，分配 Event_id
│   ├── io/                          # I/O 与格式约定
│   │   ├── __init__.py
│   │   ├── schema.py                # 列名、dtype、必填字段定义
│   │   ├── readers.py               # 读取 raw CSV
│   │   └── writers.py               # 写出 processed CSV
│   ├── utils/                       # 通用工具（几何、时间、ID 等）
│   │   ├── __init__.py
│   │   ├── geometry.py              # 相对位置、距离、车道/方位判断
│   │   └── time_utils.py            # 时间戳对齐、秒↔帧换算
│   └── scripts/                     # 对外入口脚本（调用不同工具 / 跑全流程）
│       ├── run_pipeline.py          # 端到端：raw → processed
│       ├── run_conflict_detect.py   # 单独跑冲突检测
│       ├── run_window_check.py      # 单独跑轨迹窗口校验
│       └── run_neighbor_extract.py  # 单独跑周边车采集
│
├── model_baseline/                  # ========== 模块二：模型 Baseline ==========
│   ├── __init__.py
│   ├── dataset.py                   # 读取 processed data/label，构造 Dataset
│   ├── models/                      # 模型定义占位
│   │   └── baseline.py
│   ├── train.py
│   ├── evaluate.py
│   └── scripts/
│       ├── run_train.py
│       └── run_eval.py
│
└── tests/                           # 单测 / 小样例回归（可选）
    ├── data_processing/
    └── model_baseline/
```

### 分层约定

- **`core/`**：纯逻辑函数，尽量无 CLI、少副作用，便于单测与复用。
- **`scripts/`**：命令行入口，解析配置与路径，调用 `core` / `pipeline`。
- **`pipeline.py`**：定义阶段顺序与数据在阶段间的流转对象，是数据处理主线的“说明书式代码”。
- **`io/schema.py`**：唯一真相源（Single Source of Truth）——原始列、中间结构、输出 data/label 列在此集中声明。

---

## 3. 数据处理主线流程

### 3.1 流程概览

```text
Raw CSV (标准化轨迹)
        │
        ▼
┌───────────────────────┐
│ 1. Conflict 检测       │  标记 conflict / non-conflict 候选时刻或区间
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 2. 轨迹窗口校验        │  history=8s；conflict 落在未来 3s 区间内；
│                       │  以该 3s 区间起点为 t0，向前取 8s 作为输入窗口
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 3. 周边车筛选与采集    │  在 [t0-8s, t0]（及后续约定时长）内，
│                       │  凡进入过 ego 邻域的目标车持续采轨迹；
│                       │  方位：前 / 后 / 左前 / 左后 / 右前 / 右后；
│                       │  距离 > 200m 不纳入
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 4. 导出 Data + Label   │  继承原 CSV 基础格式 + Event_id；
│                       │  data 与 label 分文件，由 Event_id 关联
└───────────────────────┘
```

### 3.2 时间窗口约定

| 符号 | 含义 | 时长 |
|------|------|------|
| `t0` | 预测起点 / 观测终点 | — |
| History 窗口 | `[t0 - 8s, t0]` | 8s |
| Future / Label 窗口 | `[t0, t0 + 3s]` | 3s |
| Conflict 出现位置 | 落在 Future 窗口内（可随机/按规则采样） | ⊆ 3s |

**Conflict 样本：**

1. 先定位冲突发生时刻 `t_conflict`（或冲突区间）。
2. 要求 `t_conflict ∈ [t0, t0 + 3s]`。
3. 校验 `[t0 - 8s, t0]` 内 ego（及所需车辆）轨迹完整、可对齐。
4. 不满足长度/完整性则丢弃该候选。

**Non-conflict 样本：**

1. 在无冲突区间上按同样窗口规则采样 `t0`。
2. 同样要求 history 8s 完整；label 侧标记无冲突。

### 3.3 周边车规则

在**目标 duration**（至少覆盖 history 窗口；实现时可配置是否延伸到 future）内：

1. 以 ego 为中心，按相对方位划分为：
   - `front` / `rear`
   - `left_front` / `left_rear`
   - `right_front` / `right_rear`
2. **进入过**上述邻域的车辆，在整个 duration 内**持续采集**其轨迹（不是只采“当前帧在邻域内”的片段）。
3. 与 ego 距离 **> 200m** 的观测不纳入（或整车不纳入，具体策略在配置中明确：按帧剔除 vs 按车剔除）。
4. 输出中保留车辆角色/槽位信息，便于模型对齐多车输入。

### 3.4 阶段间流通对象（逻辑结构，非必须落盘）

```text
RawTrajectoryTable
  → ConflictCandidate[]          # 含 scene_id, t_conflict / is_conflict, ego_id, ...
  → WindowedEventCandidate[]     # 含 t0, history_range, future_range, 校验通过标记
  → TrackedNeighborhoodEvent[]   # 含 ego + neighbors 在窗口内的轨迹表
  → (EventDataRows, EventLabelRow)
```

中间结构体定义在 `data_processing/io/schema.py`：
`ConflictCandidate` → `WindowedEventCandidate` → `TrackedNeighborhoodEvent` → CSV。

主流程编排见 `data_processing/pipeline.py`：

```text
process_raw_dataframe / run_pipeline
  run_conflict_stage   -> detect_all_candidates
  run_window_stage     -> build_windowed_events
  run_neighbor_stage   -> extract_neighbors_for_events
  run_export_stage     -> export_events
```

---

## 4. 函数 API 清单（已落在代码中的签名）

> 算法体多为 `NotImplementedError` 占位；名称与调用链已固定，便于分步实现。

### 4.1 `core/conflict_detect.py`

| 函数 | 作用 |
|------|------|
| `detect_conflicts(raw_df, cfg)` | 检测 conflict 候选 |
| `sample_non_conflicts(raw_df, cfg, exclude=...)` | 采样 non-conflict 候选 |
| `detect_all_candidates(raw_df, cfg)` | 合并上述两类 |

### 4.2 `core/trajectory_window.py`

| 函数 | 作用 |
|------|------|
| `propose_t0_for_conflict(candidate, cfg)` | 使 `t_conflict ∈ [t0, t0+3s]` 并回推 8s |
| `propose_t0_for_non_conflict(candidate, cfg)` | non-conflict 的 t0 提案 |
| `validate_history_length(...)` | 校验 history 8s 完整性 |
| `validate_future_interval(...)` | 校验 future 3s / 冲突落点 |
| `build_windowed_candidate(...)` | 单候选 → Windowed 或丢弃 |
| `build_windowed_events(...)` | 批量窗口构建 |

### 4.3 `core/neighbor_filter.py`

| 函数 | 作用 |
|------|------|
| `find_entered_neighbor_ids(...)` | duration 内进入过邻域的车 |
| `assign_neighbor_roles(...)` | 分配六方位 slot |
| `collect_persistent_tracks(...)` | 进入过则全程持续采轨迹 |
| `extract_neighbors_for_event(...)` | 单事件邻车流水线 |
| `extract_neighbors_for_events(...)` | 批量 |

### 4.4 `core/export.py`

| 函数 | 作用 |
|------|------|
| `assign_event_ids(events)` | 分配 Event_id |
| `build_event_data_rows(...)` | 构造 history data 表 |
| `build_event_label_row(...)` | 构造单行 label |
| `build_future_traj_rows(...)` | 构造 future 3s 轨迹表 |
| `events_to_tables(...)` | → (data_df, label_df, future_df) |
| `export_events(...)` | 写三个 CSV |

### 4.5 `utils/` / `io/`

| 模块 | 函数 |
|------|------|
| `utils/geometry.py` | `compute_distance`, `compute_relative_pose`, `classify_neighbor_slot`, `is_within_max_distance` |
| `utils/time_utils.py` | `sec_to_frames`, `frames_to_sec`, `get_track_time_span`, `has_continuous_coverage`, `to_relative_time`, `window_bounds` |
| `io/readers.py` | `load_config`, `load_raw_csv`, `list_raw_csv_files` |
| `io/writers.py` | `write_events_data`, `write_events_labels`, `write_future_traj`, `write_interim_candidates` |
| `io/schema.py` | 列契约 + `ProcessingConfig` + 中间 dataclass |

### 4.6 脚本入口

| 脚本 | 调用 |
|------|------|
| `scripts/run_pipeline.py` | `run_pipeline` 全流程 |
| `scripts/run_conflict_detect.py` | `detect_all_candidates` → interim CSV |
| `scripts/run_window_check.py` | detect → `build_windowed_events` |
| `scripts/run_neighbor_extract.py` | detect → window → `extract_neighbors_for_events`（可选 export） |

---

## 5. 流通数据格式设计

### 5.1 原始输入（Raw）——标准化 CSV

保持项目既有标准化列（示意；以 `schema.py` 最终定义为准）：

| 类别 | 示例字段 |
|------|----------|
| 场景 / 帧 | `scene_id`, `frame_id`, `timestamp` |
| 物体 | `track_id`, `agent_type` |
| 状态 | `x`, `y`, `vx`, `vy`, `ax`, `ay`, `heading`, ... |
| 其他 | 车道、尺寸等（若已有则原样保留） |

原始文件中通常**尚无** `Event_id`；事件在处理后生成。

### 5.2 处理后 Data（`data/processed/data/events_data.csv`）

**原则：** 尽量沿用 raw 基础列，增加事件与角色字段；一行 = 某一 `Event_id` 下某一时刻某一车辆的一条轨迹点。

| 字段 | 说明 |
|------|------|
| `Event_id` | 事件序号（全局递增或按 scene 内递增，配置决定） |
| `scene_id` | 来源场景 |
| `timestamp` / `frame_id` | 时间 |
| `track_id` | 车辆 ID |
| `role` | `ego` / `front` / `rear` / `left_front` / `left_rear` / `right_front` / `right_rear` / `other`（若需要） |
| `x`, `y`, `vx`, `vy`, ... | 与 raw 一致的运动学列 |
| `t_rel` | 相对 `t0` 的时间（s），history 为负、`t0` 为 0（推荐，便于训练对齐） |
| （可选）`split` | train/val/test |

**范围建议：** Data 文件主要存放 **history 8s**（以及配置允许的 neighbor 持续轨迹片段）。Future 3s 可仅放在 label，或在 data 中用 `t_rel > 0` 区分——推荐 **future 只进 label**，避免泄漏与重复。

### 5.3 处理后 Label（`data/processed/labels/events_labels.csv`）

**原则：** 一行 = 一个 `Event_id` 的监督信息；未来轨迹可扁平化或多行子表（二选一，推荐主表 + 可选 future 轨迹表）。

#### 方案 A（推荐）：主标签表 + 未来轨迹表

**`events_labels.csv`（每事件一行）：**

| 字段 | 说明 |
|------|------|
| `Event_id` | 与 data 关联 |
| `scene_id` | 冗余便于排查 |
| `t0` | 预测起点时间戳 |
| `is_conflict` | `0` / `1` |
| `conflict_target_id` | 冲突对手 `track_id`；非冲突为空白或 `-1` |
| `conflict_target_role` | 可选：对手当时/主要方位槽位 |
| `t_conflict` | 可选：冲突时刻（相对或绝对） |

**`events_future_traj.csv`（每事件多行）：**

| 字段 | 说明 |
|------|------|
| `Event_id` | 关联 |
| `track_id` | 通常含 ego，及需要预测/对照的车 |
| `timestamp` / `t_rel` | ∈ `(0, 3s]` |
| `x`, `y`, ... | 未来 3s 轨迹（与 raw 同列风格） |

#### 方案 B：单文件宽表

将 `is_conflict`、`conflict_target_id` 与未来轨迹序列序列化进同一 CSV（如 JSON 字符串列）。实现简单，但不如方案 A 清晰，**不优先**。

### 5.4 关联关系

```text
events_data.csv          ──Event_id──►  events_labels.csv
       │                                      │
       │                                      └──Event_id──► events_future_traj.csv
       └── 仅 history（及邻车持续轨迹）
```

### 5.5 Event_id 规则（建议）

- 处理流水线成功导出一个样本时分配一个新 `Event_id`。
- 建议格式：`{scene_id}_{seq:06d}` 或全局整数 `0..N-1`。
- 同一 `Event_id` 下 data 与 label **必须**一一对应；丢弃样本不得留下半边文件。

---

## 6. 配置项清单（`configs/data_processing.yaml` 示意）

```yaml
time:
  history_sec: 8.0
  future_sec: 3.0
  # conflict 必须落在 future 窗口内

neighbor:
  max_distance_m: 200.0
  slots: [front, rear, left_front, left_rear, right_front, right_rear]
  # persist_policy: once_entered_keep_for_full_duration

io:
  raw_dir: data/raw
  data_out: data/processed/data/events_data.csv
  label_out: data/processed/labels/events_labels.csv
  future_traj_out: data/processed/labels/events_future_traj.csv
```

---

## 7. 模型 Baseline 模块边界

`model_baseline/` **只消费** `data/processed/` 下的 data + label：

1. `dataset.py`：按 `Event_id` join，组装张量（多车槽位对齐、时间对齐）。
2. `models/baseline.py`：占位网络结构。
3. `train.py` / `evaluate.py`：训练与指标（冲突分类 + / 或轨迹预测，按挑战定义再定）。

数据处理与模型训练通过 **磁盘上的 CSV 契约** 解耦，避免互相 import 内部实现。

---

## 8. 脚本职责对照

| 脚本 | 作用 |
|------|------|
| `data_processing/scripts/run_pipeline.py` | 跑完整 1→4 流程 |
| `data_processing/scripts/run_conflict_detect.py` | 只跑检测，可写 interim |
| `data_processing/scripts/run_window_check.py` | 只跑窗口校验 |
| `data_processing/scripts/run_neighbor_extract.py` | 只跑邻车采集 |
| `model_baseline/scripts/run_train.py` | 训练入口 |
| `model_baseline/scripts/run_eval.py` | 评估入口 |

核心算法放在 `data_processing/core/*`，脚本只负责参数与调用。

---

## 9. 后续实现顺序建议

1. 冻结 `io/schema.py`（raw / data / label 列）。
2. 实现 `conflict_detect` → `trajectory_window` → `neighbor_filter` → `export`。
3. 用 `run_pipeline.py` 打通小样例 CSV。
4. 再启动 `model_baseline/dataset.py` 对接 processed 产物。

---

## 10. 设计要点小结

- **两模块分文件夹**：`data_processing/` 与 `model_baseline/`。
- **逻辑与入口分离**：`core/` 写主逻辑，`scripts/` 调工具与流水线。
- **时间规则**：输入 8s history；冲突落在随后 3s；以 3s 区间起点为 `t0` 回推 8s。
- **邻车**：六方位 + 200m；duration 内进入过则全程持续采集。
- **存储**：CSV；保留原格式列并加 `Event_id`；**data 与 label 分开放**。
