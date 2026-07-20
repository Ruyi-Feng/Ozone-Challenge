# Ozone-Challenge 开发进度

**最后更新**: 2026-07-20  
**仓库**: Ruyi-Feng/Ozone-Challenge  
**分支**: main (本地)  
**Git 提交**: 6 个 commit

---

## 一、项目概述

将原始 NBDT 标准化轨迹 CSV 处理为模型训练用的 data + label 样本集。

```
原始轨迹 CSV → [Stage 1: 冲突检测] → [Stage 2: 窗口校验] → [Stage 3: 周边车] → [Stage 4: 导出] → data/label CSV
```

**核心参数**：history 8s (200帧), future 3s (75帧), fps=25, TTC阈值=3s, 距离上限=200m

---

## 二、当前完成状态

### ✅ 数据处理流水线 — 全部完成

| 阶段 | 文件 | 关键内容 | 状态 |
|------|------|---------|:---:|
| — | `utils/geometry.py` | 距离、相对位姿、六方位 slot 分类 | ✅ |
| — | `utils/time_utils.py` | 秒帧转换、连续覆盖检查、窗口边界 | ✅ |
| — | `io/schema.py` | 列契约、中间数据结构、ProcessingConfig | ✅ |
| — | `io/readers.py` | 读 YAML 配置、读 CSV + schema 校验 | ✅ |
| — | `io/writers.py` | 写三类输出 CSV + interim | ✅ |
| — | `pipeline.py` | 主线编排 (raw → 四阶段 → CSV) | ✅ |
| Stage 1 | `core/conflict_detect.py` | NBDT 标准 2D_TTC 冲突检测，18 测试通过 | ✅ |
| Stage 2 | `core/trajectory_window.py` | t0 提案、8s+3s 窗口校验 | ✅ |
| Stage 3 | `core/neighbor_filter.py` | 六方位邻车筛选、persistent track | ✅ |
| Stage 4 | `core/export.py` | Event_id 分配、data/label/future CSV | ✅ |
| Scripts | `scripts/run_pipeline.py` 等 4 个 | CLI 入口 | ✅ |
| Config | `configs/data_processing.yaml` | fps, TTC阈值, 距离等参数 | ✅ |
| Docs | `docs/REPO_STRUCTURE.md` | 架构设计文档 | ✅ |
| Docs | `docs/METHODOLOGY.md` | 完整方法论 | ✅ |
| Tests | `tests/.../test_conflict_detect_fix.py` | 18 个测试用例 | ✅ |

### ❌ 模型模块 — 全部 TODO

| 文件 | 状态 |
|------|:---:|
| `model_baseline/dataset.py` | TODO |
| `model_baseline/models/baseline.py` | TODO |
| `model_baseline/train.py` | TODO |
| `model_baseline/evaluate.py` | TODO |
| `configs/model_baseline.yaml` | TODO |

### ❌ 数据 — 空

| 目录 | 状态 |
|------|:---:|
| `data/raw/` | 空（待放 NBDT 标准 CSV） |
| `data/interim/candidates/` | 空 |
| `data/processed/` | 空 |

---

## 三、Stage 1 冲突检测 — 完整方法论

### 2D_TTC 计算流程

```
对每帧的每对 (ego, target)，target 在 ego 前方六方位内:

  1. _calculate_nearest_points()      SAT碰撞检测 → OBB最近点距离
  2. _rear_forward_strips_intersect() 4种几何检查 → 前后碰撞姿态?
  3. _compute_2d_ttc_kernel()         速度+加速度投影，角速度修正
                                     解 dist = closing·t + ½a·t²
                                     → 取最小正根 = 2D_TTC

  2D_TTC < 3s → 冲突帧
  连续冲突帧 → 聚合为事件 → ConflictCandidate
```

### 对齐 NBDT 标准

以下 12 个函数逐行对齐 NBDT `ssm.py`：

| 函数 | NBDT 对应 |
|------|----------|
| `_order_rect_points` | `GeometryHelper.order_rect_points` |
| `_line_segment_intersection` | `GeometryHelper.line_segment_intersection` |
| `_is_inside_rect` | `GeometryHelper._is_inside_rect` |
| `_rectangles_intersect` | `GeometryHelper._rectangles_intersect` |
| `_calculate_nearest_points` | `GeometryHelper.calculate_nearest_points` |
| `_rear_edge_from_ordered_bbox` | `InstantSSMCalculator._rear_edge_from_ordered_bbox` |
| `_point_in_forward_strip` | `InstantSSMCalculator._point_in_forward_strip` |
| `_ray_intersect_segment` | `InstantSSMCalculator._ray_intersect_segment` |
| `_two_rays_intersect_forward` | `InstantSSMCalculator._two_rays_intersect_forward` |
| `_rear_forward_strips_intersect` | `InstantSSMCalculator._rear_forward_strips_intersect` |
| `_project_to_line` | `InstantSSMCalculator._project_to_line` |
| `_compute_2d_ttc_kernel` | `InstantSSMCalculator._compute_2d_ttc_kernel` |

### 修复历史

**第一版 Bug**：`_strips_intersect` 只做一条线交叉+纵向范围检查，两年辆同向时前后边缘平行→无交点→全部返回 None→0 个 conflict。

**第一版修复**：用 `_is_front_to_rear_contact`（OBB 最近点半区检测）替代，10 测试通过。

**第二版（当前）**：对齐 NBDT 标准，完整实现四检查 strip 交叉 + 二次方程 TTC + 加速度差分，18/18 测试通过。

---

## 四、输出格式

### 一个 Event 包含的信息

```
Event_id="3"
├── ego_id=42, scene_id="scene_01"
├── t0=200.0（预测起点帧号）
├── is_conflict=1, conflict_target_id=57, conflict_target_role="front", t_conflict=206.0
├── 参与者：ego(42), front(57), left_front(63), right_rear(18), left_rear(91)
├── events_data.csv: 5辆车 × 200帧 ≈ 1000行 history 轨迹
└── events_future_traj.csv: 5辆车 × 75帧 future 轨迹（可选）
```

### 三个输出 CSV

| 文件 | 粒度 | 内容 |
|------|------|------|
| `events_data.csv` | 每行=某Event某车某一帧 | history 8s 轨迹（t_rel ≤ 0），含 role 列 |
| `events_labels.csv` | 每行=一个Event | is_conflict, conflict_target_id, t0, t_conflict |
| `events_future_traj.csv` | 每行=某Event某车某一帧 | future 3s 轨迹（t_rel > 0），可选 |

通过 Event_id 关联。含冲突和非冲突两类样本（等量采样）。

---

## 五、Git 提交记录

```
8bb6661 feat: align 2D_TTC with NBDT standard implementation         ← HEAD
         data_processing/core/conflict_detect.py           +564/-300
         tests/.../test_conflict_detect_fix.py             +315/-?  (18 tests)

ec7f2ab fix: replace strip-intersection with front/rear contact check
         data_processing/core/conflict_detect.py           +463/-17
         tests/.../test_conflict_detect_fix.py             +260     (first version, 10 tests)

8d9123c feat: implement utils, trajectory_window, neighbor_filter, export
         8 files, +532/-79 (Stages 2-4 + utils + io update)

22f7c8b fix: ban data (.gitignore)

e8e301a feat: init structure of repo (36 files, +1478)

81e4d88 Initial commit
```

---

## 六、下一步

1. ❌ 获取/放入真实 NBDT 标准化轨迹 CSV 到 `data/raw/`
2. ❌ 小样本跑全流水线验证
3. ❌ 实现 `model_baseline/` 模块
4. ❌ Push 到远程仓库
5. ❌ `docs/METHODOLOGY.md` 和 `PROGRESS.md` 未提交
