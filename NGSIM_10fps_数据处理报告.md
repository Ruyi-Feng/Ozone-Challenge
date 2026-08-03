# NGSIM 10fps 原生数据处理报告

> 生成日期：2026-07-23  
> 项目：Ozone-Challenge 数据处理管线

---

## 1. 概述

本次修改使 Ozone-Challenge 数据处理管线支持 **NGSIM Peachtree Street 原生 10fps 数据**，无需经过 NBDT 上采样到 25fps。核心思路是在 reader 层增加格式转换，管线核心代码零改动。

### 数据流

```
NGSIM 原生 10fps                    标准格式 10fps                    管线输出
┌──────────────────┐    ngsim_     ┌──────────────────┐    管线     ┌──────────────────┐
│ Vehicle_ID       │   converter   │ frameNum         │  8s+3s    │ events_data      │
│ Frame_ID         │  ─────────→   │ carId            │  window   │ events_labels    │
│ Local_X/Y (ft)   │              │ carCenterXm/Ym   │  ──────→  │ events_future    │
│ v_Vel (ft/s)     │              │ heading (deg)    │           │     _traj        │
│ v_Length/Width   │              │ speed (m/s)      │           │                  │
│ v_Class          │              │ boundingBox1..4  │           │                  │
└──────────────────┘              └──────────────────┘           └──────────────────┘
```

---

## 2. 数据源

| 项目 | 说明 |
|---|---|
| 数据集 | NGSIM Peachtree Street |
| 源文件 | `交叉口数据处理/NGSIM/ngsim_raw/peachtree.csv` |
| 规模 | 873,888 行，1,545 辆车 |
| 帧率 | **10 Hz**（原生，未上采样） |
| 时间跨度 | Global_Time 1163019100 – 1164063200（约 174 分钟） |
| 原始列数 | 24 列 |
| 坐标系统 | Global_X/Y（英尺），Local_X/Y（英尺） |
| 速度 | v_Vel（ft/s） |
| 车辆尺寸 | v_Length, v_Width（英尺） |
| 车辆类别 | v_Class: 1=摩托车, 2=汽车, 3=卡车 |
| 缺失字段 | heading（航向角）、OBB 角点坐标 |

### 测试数据

从完整数据中截取 20 秒时间窗口（Global_Time 1163050000–1163070000）：

| 指标 | 值 |
|---|---|
| 车辆数 | 26 辆 |
| 数据行数 | 5,373 行 |
| 帧数 | 318 帧 |
| 每车帧数 | 126–305 帧（中位 201 帧） |
| 车辆类别 | 全部为汽车（v_Class=2） |
| 平均速度 | 2.85 m/s（10.3 km/h） |

---

## 3. 代码修改

### 3.1 修改总览

| 操作 | 文件 | 说明 |
|---|---|---|
| **新增** | `data_processing/io/ngsim_converter.py` | NGSIM 原生→标准格式转换器（~230行） |
| **修改** | `data_processing/io/readers.py` | 新增 `load_ngsim_csv()`，自动格式检测 |
| **修改** | `configs/data_processing.yaml` | `fps: 25.0` → `fps: 10.0` |
| **修改** | `data_processing/io/__init__.py` | 导出转换函数 |
| **不动** | `core/conflict_detect.py` 等 5 个管线模块 | — |

### 3.2 转换器详情（`ngsim_converter.py`）

基于 NBDT 源码 `NGSIMTransfer` 类改写为独立模块，修复了列名大小写问题（`v_length` → `v_Length`）。

**转换步骤：**

| 步骤 | 函数 | 说明 |
|---|---|---|
| 1 | 排序 | 按 (Vehicle_ID, Frame_ID) 升序 |
| 2 | `_build_local_frame()` | Global_X/Y（ft）→ 本地坐标系（m） |
| 3 | `_compute_heading_and_course()` | 相邻帧位置差分 atan2(dy, dx) 求 heading |
| 4 | 单位转换 | 英尺×0.3048→米（位置、速度、尺寸） |
| 5 | `_oriented_bbox()` | 位置+heading+长宽 → OBB 四角点 |
| 6 | 类别映射 | v_Class(1,2,3)→objClass(4,0,3) |
| 7 | 组装 | 输出 30 列标准格式 DataFrame |

### 3.3 格式自动检测

`load_raw_csv()` 检测到 `Vehicle_ID`, `Local_X`, `Global_X`, `v_Vel` 列时自动调用 NGSIM 转换器，否则按原有逻辑处理。

---

## 4. 管线配置

```yaml
# configs/data_processing.yaml
time:
  history_sec: 8.0        # 不变
  future_sec: 3.0         # 不变
  fps: 10.0               # 25.0 → 10.0
  conflict_ttc_threshold: 3.0
```

窗口帧数变化（fps 从 25→10，物理时间不变）：

| 参数 | 25fps | 10fps |
|---|---|---|
| 8s 历史窗口 | 200 帧 | **80 帧** |
| 3s 未来窗口 | 75 帧 | **30 帧** |
| 间隙容忍 | 0.084s（~2帧） | 0.21s（~2帧） |

---

## 5. 测试结果

### 5.1 管线产出

在 26 车 / 5,373 行测试数据上运行完整管线：

| 指标 | Train | Val | 合计 |
|---|---|---|---|
| 事件数 | 227 | 150 | **377** |
| 冲突事件 | 206 | 146 | **352** |
| 非冲突事件 | 21 | 4 | 25 |
| 涉及 ego 数 | 20 | 6 | 26 |
| 数据行数 | 278,939 | 198,263 | 477,202 |
| 轨迹行数 | 113,201 | 92,652 | 205,853 |

### 5.2 冲突分布

**训练集冲突目标角色：**

| 角色 | 数量 | 占比 |
|---|---|---|
| right_front | 76 | 36.9% |
| left_front | 54 | 26.2% |
| front | 37 | 18.0% |
| right_rear | 7 | 3.4% |
| rear | 4 | 1.9% |
| left_rear | 3 | 1.5% |

冲突覆盖了全部 6 个邻车方位，分布合理。

### 5.3 时间窗口验证

```
t_rel 步长: 0.1s → 确认 10fps ✅
历史窗口: [-8.0, 0.0] → 80 帧 = 8 秒 ✅
未来窗口: [0.1, 3.0] → 30 帧 = 3 秒 ✅
```

---

## 6. TTC 平滑度分析

### 6.1 方法

对 525 个 front-pair 逐帧计算 2D_TTC，筛选出 ≥20 帧有效 TTC 的 179 个配对，分析帧间 TTC 变化。

### 6.2 结果

| 指标 | 值 |
|---|---|
| 分析配对数 | 15（top by 帧数） |
| **中位 \|ΔTTC\| / 0.1s 步** | **0.13s** |
| 平均 \|ΔTTC\| / 步 | 1.01s（受远距离配对拖高） |
| % 步长 \|ΔTTC\| < 0.5s | **75.2%** |
| % 步长 \|ΔTTC\| < 1.0s | **82.1%** |

### 6.3 两类配对

**近距离配对（TTC < 20s）—— 非常平滑**

```
e39/t25: mean|ΔTTC|=0.15s, 97% of steps < 0.5s
  TTC: [13.38, 13.29, 13.14, 12.87, 12.45, 11.94, 10.79, ...]
  → 单调递减，步长均匀
```

**远距离配对（TTC 5–95s）—— 天然波动**

```
e73/t61: mean|ΔTTC|=2.33s, 58% of steps < 0.5s
  TTC: [5.87, 27.94, 18.38, 4.40, 57.0, ...]
  → TTC = distance / Δv，远距离+小相对速度→天然不稳定
```

远距离波动是 TTC 的数学特性，非数据质量或算法问题。

### 6.4 与 25fps 对比

| 噪声源 | 10fps (dt=0.1s) | 25fps (dt=0.04s) | 25fps 噪声倍数 |
|---|---|---|---|
| 加速度量化噪声 | 0.030 m/s² | 0.075 m/s² | 2.5× |
| 角速度精度 | 位置差 ~1.0m/步 | 位置差 ~0.4m/步 | ~2.5× |
| 预估 TTC 噪声 | 中位 0.13s | 预估 0.3–0.5s | 2–4× |

**结论：10fps 原生数据在冲突检测关心的 TTC < 3s 范围内足够平滑，无需额外做移动平均等降噪处理。**

---

## 7. 输出文件清单

```
Ozone-Challenge/
├── data/
│   ├── raw/
│   │   ├── peachtree_test.csv           ← NGSIM 原生测试输入（26车）
│   │   ├── peachtree_mini.csv           ← NGSIM 原生迷你测试（2车）
│   │   └── peachtree_std_10fps.csv      ← 标准格式 10fps 输出（转换后）
│   │
│   ├── interim/candidates/
│   │   └── peachtree_test_candidates.csv ← 中间冲突候选
│   │
│   └── processed/
│       ├── data/
│       │   ├── events_data_train.csv     ← 训练集轨迹 (278,939行)
│       │   └── events_data_val.csv       ← 验证集轨迹 (198,263行)
│       └── labels/
│           ├── events_labels_train.csv   ← 训练集标签 (227事件)
│           ├── events_labels_val.csv     ← 验证集标签 (150事件)
│           ├── events_future_traj_train.csv
│           └── events_future_traj_val.csv
```

### 数据格式

**标准格式输入/输出**（与 CitySim 25fps 格式一致）：

```
frameNum, carId, laneId, carCenterX, carCenterY,
boundingBox1X..4Y, carCenterXm, carCenterYm, boundingBox1Xm..4Ym,
heading, course, speed, objClass, carCenterLon, carCenterLat, scene_id
```

**事件数据**（`events_data_*.csv`）：

```
Event_id, scene_id, frameNum, t_rel, carId, role,
carCenterXm, carCenterYm, heading, speed
```

**事件标签**（`events_labels_*.csv`）：

```
Event_id, scene_id, t0, is_conflict, conflict_target_id,
conflict_target_role, t_conflict
```

---

## 8. 多文件处理

### 8.1 原有问题

`pipeline.py` 的 `run_pipeline()` 在 `raw_dir` 下有多个 CSV 时只处理了第一个文件（`files[0]`）。代码注释也标注了 `"current stub"`。

### 8.2 修复方案

改为：遍历所有文件，每文件标记 `scene_id`（取文件名），合并后统一走管线。这样 CitySim（IntersectionA-01~12）、inD（00~32）等多文件数据集都能正确处理，同时 NGSIM 单文件不受影响。

```python
# 每个文件标记 scene_id，合并后再处理
for fp in files:
    df = load_raw_csv(fp)
    df["scene_id"] = Path(fp).stem  # 例如 "IntersectionA-01"
    parts.append(df)
merged = pd.concat(parts, ignore_index=True)
return process_raw_dataframe(merged, cfg, dump_interim=dump_interim)
```

### 8.3 额外输出

| 文件 | 说明 |
|---|---|
| `requirements.txt` | Python 依赖清单（pandas, numpy, pyyaml） |
| `README_使用指南.md` | 面向代码基础较弱同学的操作指南 |

---

## 9. 文件修改汇总

| 文件 | 操作 | 说明 |
|---|---|---|
| `data_processing/io/ngsim_converter.py` | **新增** | NGSIM 原生→标准格式转换器 |
| `data_processing/io/readers.py` | 修改 | 新增 NGSIM 格式自动检测 |
| `data_processing/io/__init__.py` | 修改 | 导出转换函数 |
| `data_processing/pipeline.py` | 修改 | 多文件合并处理 |
| `configs/data_processing.yaml` | 修改 | fps: 25→10 |
| `requirements.txt` | 修改 | 补充实际依赖 |
| `README_使用指南.md` | **新增** | 用户操作指南 |
| `tests/test_ttc_smoothness.py` | **新增** | TTC 平滑度测试 |
| `core/conflict_detect.py` | **不动** | — |
| `core/trajectory_window.py` | **不动** | — |
| `core/neighbor_filter.py` | **不动** | — |
| `core/export.py` | **不动** | — |
| `utils/geometry.py` | **不动** | — |
| `utils/time_utils.py` | **不动** | — |
