# Ozone-Challenge 数据处理管线 — 使用指南

> 写给不太熟悉代码的同学：跟着步骤走就能跑通。

---

## 一、环境准备（只需做一次）

### 1. 确认 Python 版本

打开命令行（Win+R 输入 `cmd`），输入：

```bash
python --version
```

需要 **Python 3.10 或以上**。如果版本太低，去 [python.org](https://www.python.org) 下载安装。

### 2. 安装依赖

在项目根目录下运行（就是有 `requirements.txt` 的那个文件夹）：

```bash
cd D:\研究实习\暑期实习\项目架构\Ozone-Challenge
pip install -r requirements.txt
```

依赖很少，只有三个：pandas、numpy、pyyaml。安装很快。

---

## 二、准备数据

### 把数据放在哪

把你的 CSV 数据文件放到这个文件夹：

```
data\raw\
```

**两种数据格式都可以直接放进去，管线会自动识别：**

#### 格式一：k原生格式（10fps）

列名类似这样（从 peachtree 导出的原始数据）：

```
Vehicle_ID, Frame_ID, Local_X, Local_Y, Global_X, Global_Y,
v_Length, v_Width, v_Class, v_Vel, v_Acc, Lane_ID, ...
```

#### 格式二：标准格式（NBDT 标准化）

列名类似这样：

```
frameNum, carId, carCenterXm, carCenterYm, heading, speed,
objClass, boundingBox1Xm, boundingBox1Ym, ...
```

> **不需要手动区分**，管线会自动检测格式并做转换。NGSIM 原生数据会自动转为标准格式（保持 10fps）。

### 示例文件结构

```
data\raw\
  ├── peachtree.csv              ← NGSIM 原生 10fps（单文件）
  ├── IntersectionA-01.csv       ← CitySim 标准格式（多文件示例）
  ├── IntersectionA-02.csv
  └── ...
```

---

## 三、运行管线

### 最简单的方式（处理 raw 文件夹下所有 CSV）

```bash
python -m data_processing.scripts.run_pipeline --config configs/data_processing.yaml
```

这会自动读取 `data/raw/` 下所有 CSV，合并处理，输出结果。

### 单独处理一个文件

```bash
python -m data_processing.scripts.run_pipeline --config configs/data_processing.yaml --raw data/raw/peachtree.csv
```

### 如果数据是 10fps 格式

什么都不用改，管线自动识别。配置文件里的 `fps: 10.0` 已经设好了。



---

## 四、输出在哪

运行成功后，输出在 `data/processed/` 下：

```
data/processed/
├── data/
│   ├── events_data_train.csv     ← 训练集：每条轨迹的每一帧
│   └── events_data_val.csv       ← 验证集：每条轨迹的每一帧
└── labels/
    ├── events_labels_train.csv    ← 训练集标签：是否冲突、和谁冲突
    ├── events_labels_val.csv      ← 验证集标签
    ├── events_future_traj_train.csv  ← 训练集：未来 3 秒轨迹
    └── events_future_traj_val.csv    ← 验证集：未来 3 秒轨迹
```

### 输出文件字段说明

**events_data：** 每一行是一帧

| 列名 | 说明 |
|---|---|
| Event_id | 事件编号 |
| scene_id | 场景（数据文件名） |
| frameNum | 帧号 |
| t_rel | 相对 t0 的时间（秒），历史为负，未来为正 |
| carId | 车辆 ID |
| role | ego（主车）/ front / rear / left_front 等 |
| carCenterXm, carCenterYm | 车辆中心坐标（米） |
| heading | 航向角（度，0=东，90=北） |
| speed | 速度（m/s） |

**events_labels：** 每一行是一个事件

| 列名 | 说明 |
|---|---|
| Event_id | 事件编号 |
| scene_id | 场景 |
| t0 | 预测起始帧 |
| is_conflict | 是否冲突（1=是，0=否） |
| conflict_target_id | 冲突对象车辆 ID（非冲突时为 -1） |
| conflict_target_role | 冲突对象的相对位置（front / left_front 等） |
| t_conflict | 冲突发生的帧号 |

---

## 五、配置说明

配置文件在 `configs/data_processing.yaml`：

```yaml
time:
  history_sec: 8.0          # 历史窗口：8 秒（不用改）
  future_sec: 3.0           # 未来窗口：3 秒（不用改）
  fps: 10.0                 # 帧率：NGSIM 用 10，CitySim 用 25
  conflict_ttc_threshold: 3.0  # TTC 冲突阈值（秒）

neighbor:
  max_distance_m: 200.0     # 邻居最大距离（米）
  slots: [front, rear, left_front, left_rear, right_front, right_rear]

split:
  train_val_ratio: 0.8      # 80% 训练 / 20% 验证
```

---

## 六、常见问题

**Q: 报错 "No CSV found under data/raw"**

A: `data/raw/` 文件夹下没有 CSV 文件。把数据放进去。

**Q: 报错 "Missing columns"**

A: CSV 列名不对。检查是否多了空格，或者不是标准格式。

**Q: 处理很慢**

A: NGSIM 全量数据（87 万行）比较慢是正常的。可以先用小样本测试：
- 截取 20 秒时间窗口
- 或者只取几辆车

**Q: 想用 NGSIM 完整数据跑**

A: 把 `peachtree.csv`（87 万行）放到 `data/raw/`，直接跑。预计需要几分钟到十几分钟。

---

## 七、项目结构速览

```
Ozone-Challenge/
├── configs/
│   └── data_processing.yaml      ← 配置文件（改参数来这里）
├── data/
│   ├── raw/                      ← ★ 把数据放这里
│   ├── interim/                  ← 中间产物（可忽略）
│   └── processed/                ← ★ 输出结果在这里
├── data_processing/              ← 管线代码（一般不用动）
│   ├── io/                       ←   读写 & 格式转换
│   ├── core/                     ←   核心处理逻辑
│   └── scripts/                  ←   命令行入口
├── requirements.txt              ← Python 依赖
└── README_使用指南.md             ← 本文档
```
