# 三种 Baseline 训练与 Shapley 归因 — 分步操作指南

> 写给要在训练机上把全流程跑通的同学。所有命令都在**仓库根目录**下执行。
>
> 本仓库的 baseline 共有三种方式(三个配置文件、三次训练):
>
> | # | 模型 | 配置文件 | 说明 |
> |---|------|----------|------|
> | ① | LSTM baseline | `configs/model_baseline.yaml` | 共享 LSTM + 双头,性能对照 |
> | ② | Transformer(无 mask) | `configs/model_transformer.yaml` | Transformer 编码器 + CLS,原始训练方式 |
> | ③ | Transformer(有 mask) | `configs/model_transformer_masked.yaml` | 掩码 surrogate 训练,**Shapley 归因的前置条件** |
>
> 只有 ③ 训练出的 checkpoint 能做 Shapley 归因(第六节);①② 的 checkpoint 对
> 掩码输入是 OOD 的,归因 driver 会直接拒绝。

---

## 一、环境准备(只做一次)

```bash
pip install -r requirements.txt      # pandas / numpy / pyyaml
pip install torch pyarrow            # 训练与缓存构建必需
pip install matplotlib               # 仅归因可视化需要
```

安装后可跑一遍测试确认环境正常(约 10 秒,应全绿):

```bash
python -m pytest tests -q
```

---

## 二、数据准备(三个模型共用)

从原始 CSV 到训练可用的缓存,共四步。如果训练机上 `data/processed/` 已有
现成缓存,跳到 **2.5 旧缓存补齐**。

### 2.1 事件抽取(CSV → 事件表)

原始数据放 `data/raw/`,然后:

```bash
python -m data_processing.scripts.run_pipeline --config configs/data_processing.yaml
```

产出 `data/processed/data/events_data_{train,val}.csv` 与
`data/processed/labels/events_labels_{train,val}.csv`。
(细节见 `README_使用指南.md`。)

### 2.2 CSV → Parquet

```bash
python data_processing/scripts/convert_csv_to_parquet.py
```

产出 `data/processed/{train,val}_events.parquet`(按 Event_id 排序的行组,
供下一步快速按事件读取)。

### 2.3 张量缓存(memmap .npy)

```bash
python data_processing/scripts/build_tensor_cache.py
```

产出(train / val 各一套):

```
data/processed/train_x.npy       float32 [N, 7, 80, 4]   多车 history
data/processed/train_mask.npy    bool    [N, 7]           slot 是否有车
data/processed/train_y.npy       float32 [N]              is_conflict
data/processed/train_target.npy  int64   [N]              冲突对象 slot
data/processed/train_eid.npy     int64   [N]              Event_id(行序基准)
data/processed/train_valid.npy   bool    [N, 7, 80]       ★ 逐帧有效性(mask 训练/归因用)
```

> `train_events.valid_events` 过滤文件是可选的:缺失时保留全部有标签事件
> (会打一行 NOTE,属正常)。
> 构建结束的 sanity 报告若提示 "N events have an all-False valid mask",
> 说明这些事件在 parquet 里没有任何真实帧,①②训练不受影响,③会自动跳过它们。

### 2.4 二进制缓存(归因必需;①②训练可跳过)

```bash
python data_processing/scripts/build_binary_cache.py
```

产出 `_data.bin` / `_index.csv` / `_valid.bin`。归因 driver(第六节)从这套
缓存读样本和元数据(event_id / scene / role / is_conflict),所以**做归因前
必须跑这一步**。

### 2.5 旧缓存补齐(训练机上已有 `_x.npy` 等文件时)

老缓存没有 `_valid.npy`。**不要重建全量缓存**(训练进程可能持有 mmap,且行序
必须保持一致),用增量模式:

```bash
python data_processing/scripts/build_tensor_cache.py --valid-only
python data_processing/scripts/build_binary_cache.py
```

`--valid-only` 以现有 `{prefix}_eid.npy` 的行序为准,单遍扫 parquet,
**只写** `_valid.npy`,不碰 x/mask/y/target。随后重跑 binary 缓存以补出
`_valid.bin`(它会整体重写 `_data.bin`/`_index.csv`,内容与之前一致)。

---

## 三、模型①:LSTM baseline

> **checkpoint 命名规则(三个模型通用)**:每个配置文件的 `model.tag` 是该
> 模型的唯一识别号,训练时自动前置到 checkpoint 文件名上——
> `checkpoints/{tag}_best_model.pt`。三个配置已各自设好 tag
> (`lstm` / `transformer` / `transformer_masked`),三次训练互不覆盖,
> 不需要手动改名。tag 留空则回到旧文件名 `best_model.pt`。

```bash
python -m model_baseline.scripts.run_train --config configs/model_baseline.yaml
```

产出 `checkpoints/lstm_best_model.pt`(每次保存时日志会打印完整路径)。

评估(不带 `--checkpoint` 时自动按配置里的 tag 找
`checkpoints/lstm_best_model.pt`):

```bash
python -m model_baseline.scripts.run_eval --config configs/model_baseline.yaml
```

> 配置里 `train.device: cuda`,无 GPU 时代码自动回退 CPU,不用改配置。

---

## 四、模型②:Transformer(无 mask)

```bash
python -m model_baseline.scripts.run_train --config configs/model_transformer.yaml
python -m model_baseline.scripts.run_eval --config configs/model_transformer.yaml
```

产出 `checkpoints/transformer_best_model.pt`(tag = `transformer`)。

这是与 ① 做预测性能对比的原始 Transformer 训练方式:全量输入、不传任何
mask、padding 帧照旧可见——与掩码改造合入前的行为逐字节一致。

---

## 五、模型③:Transformer(有 mask)

### 5.1 训练前检查清单

- [ ] `data/processed/train_valid.npy` 和 `val_valid.npy` 存在
      (2.3 全量构建自动有;旧缓存按 2.5 补)。

缺 `_valid.npy` 时训练会在启动阶段直接报错:

```
RuntimeError: train.masking.enabled=true but the train cache has no
per-frame validity mask — ... Run: python data_processing/scripts/
build_tensor_cache.py --valid-only ...
```

照报错提示补文件即可。(`masking.require_valid: false` 可以绕过,但那会把
零填充帧当真实观测进入博弈,**归因结果不可信,不要用**。)

### 5.2 启动训练

```bash
python -m model_baseline.scripts.run_train --config configs/model_transformer_masked.yaml
```

模型结构、数据、学习率与 ② 完全相同,差别只有 `model.tag`(识别号)和
配置里的 `train.masking` 段:

```yaml
train:
  masking:
    enabled: true               # 掩码 surrogate 训练开关
    p_full: 0.4                 # 40% 的样本用全量输入(保预测性能)
    seg_len_frames: [5, 10, 20] # 每次随机抽一种切段长度(0.5s/1s/2s 混合)
    hierarchies: [agent_major, time_major]   # 与归因的两种读出对应
    order_modes: [uniform, chrono, reverse]  # 与归因的排列分布对应
    require_valid: true
    val_seed: 1234              # 掩码验证用固定种子 → 跨 epoch 可比
    val_selection: mixture      # 选 checkpoint 用 混合损失(见 5.3)
```

其余 60% 样本会按"嵌套排列取随机前缀"采一个**联盟**做掩码——采样分布与第六节
Shapley 推理时的联盟分布严格一致,这正是 v(S) 良定义的关键。

### 5.3 怎么读训练日志

启动时会打一行确认掩码训练已开启:

```
Masked surrogate training ON: p_full=0.4, seg_len=(5, 10, 20), ...
```

每个 epoch 一行,比 ①② 多出 masked 验证段:

```
Epoch   3/20 | train loss=0.41 (c=0.32 t=0.09) | val loss=0.38 c_acc=0.85 t_acc=0.71 | masked val loss=0.52 c_acc=0.78 (select=0.46)
```

- `val loss / c_acc / t_acc`:**全量输入**验证(padding 帧被遮蔽)。
  这是与 ①② 对比预测性能时应引用的数字。
- `masked val loss`:固定种子随机联盟下的验证——衡量模型作为 v(S) 的质量,
  跨 epoch 可比。
- `select`:checkpoint 择优用的混合损失 = `p_full × 全量 + (1−p_full) × 掩码`。
  两种能力都好的 epoch 才会被保存。

产出 `checkpoints/transformer_masked_best_model.pt`(tag =
`transformer_masked`)。这个 checkpoint 内部带有
`train_regime: {masked: true, ...}` 标记,归因 driver 靠它识别合法的
被解释模型。

### 5.4 性能与验收

- **与 ②对比预测性能**:用训练日志中的全量 `val loss / c_acc`(而不是
  `run_eval`——run_eval 走原始 forward,padding 帧对 ③ 可见,与其训练时的
  "全量"定义略有出入,数字会有小偏差)。要求:全量输入的判别性能与 ② 持平或
  接近;明显掉点时优先调大 `p_full`(如 0.5)或加 epoch。
- **归因前验收**(4 条,依据与操作细节见 `04_轨迹WinterShapley归因方案.md` §6):
  1. 全量输入 AUC/PR-AUC 与 ② 对齐;
  2. v(∅)(全掩码)输出收敛到训练集冲突率的先验 logit;
  3. 输出随掩码率增大单调退化、无跳变;
  4. 小批事件的 P1/P4 一致性检查通过(第六节试跑时 driver 自动打印 P4 残差)。

---

## 六、Shapley 归因(模型③的后续步骤)

回答"过去 8s 轨迹里,哪些 agent 的哪些时间段、哪些特征,把未来冲突概率推高
(ψ>0)或拉低(ψ<0)"。方法与语义的完整设计见
`04_轨迹WinterShapley归因方案.md`。

### 6.1 前置条件(缺一不可)

1. **掩码训练的 checkpoint**(5.2 产出的
   `checkpoints/transformer_masked_best_model.pt`);
2. **二进制缓存含 valid**:`data/processed/val_data.bin` / `val_index.csv` /
   `val_valid.bin`(2.4 或 2.5 产出)。

### 6.2 配置

打开 `configs/attribution.yaml`,通常只需要改这几项:

```yaml
checkpoint: checkpoints/transformer_masked_best_model.pt  # 默认已指向 5.2 的产出
data_prefix: data/processed/val                      # 在验证集上做归因
device: cuda

selection:
  conflict_only: true      # 只解释冲突事件
  max_n: 100               # 先小批量;0 = 全量

game:
  seg_len_frames: 10       # 1s 一段 → 每个 agent 8 段
  readouts: [symmetric, chrono]   # 双读出;可加 reverse
  target: conflict_logit

mc:
  n_samples: 25            # 每个读出的 MC 链数;调大 → 更准更慢
  seed: 20260810

output:
  dir: outputs/attribution/run_001
```

两个读出的含义(结果解读时用):

- **symmetric(agent 主轴,均匀排列)**——"证据权重":模型的冲突判断依赖
  哪些 agent 的哪些段、哪些通道;agent 级归因由层级聚合免费得到。
- **chrono(时间主轴,时序正序)**——"风险揭示时间线":按时间顺序逐段揭示
  场景,ψ_τ 表示第 τ 段**首次带来**的风险增量,credit 记在信息最早出现的
  时刻(处理"未来依赖历史"的链状依赖)。

### 6.3 试跑(5 个事件)

```bash
python -m model_baseline.attribution.driver --config configs/attribution.yaml --max-n 5
```

控制台逐事件打印:

```
Loaded checkpoint: ... (epoch 12, masked=True)
Selected 5/812 events (conflict_only=True, max_n=5)
  [1/5] event 1042: cells=214, forwards=1830, max|P4|=3.1e-06, 2.4s
  ...
Done: 10 readout rows → outputs/attribution/run_001/summary.csv
```

检查两件事:

- `masked=True`——加载的确实是掩码 checkpoint;
- `max|P4|` 在 1e-4 量级以下——逐层 efficiency 守恒成立,归因在数学上自洽
  (它是接线正确性检查,量级异常说明实现被改动过)。

### 6.4 正式跑

确认试跑正常后放开规模:

```bash
python -m model_baseline.attribution.driver --config configs/attribution.yaml
```

也可不改 YAML 用命令行覆盖:`--checkpoint ... --data-prefix ... --max-n ...`。
耗时约每事件 1.5–5s(GPU)。同一 `mc.seed` 下结果可精确复现,且与事件
子集/顺序无关(每事件独立 RNG)。

### 6.5 输出与解读

```
outputs/attribution/run_001/
├── event_<id>.json     每事件完整结果
└── summary.csv         全部事件 × 读出的汇总行
```

`event_<id>.json` 里每个读出包含:

- `v_full` / `v_empty` / `delta_v`:全量与空集的 conflict logit 及差值
  (Σψ = delta_v,efficiency);`p_full`/`p_empty` 是对应概率,`v_empty`
  应接近训练集冲突率的先验;
- `aggregations`:cell 级 ψ 按 `agent` / `segment` / `agent_segment` /
  `channel` 四种汇总(通道 = dx/dy/heading/speed);
- `cells`:每个 (agent, 段, 通道) 单元格的 ψ 与 MC 标准误 `se`
  (|ψ| 与 se 同量级的 cell 不要过度解读,可加大 `mc.n_samples`)。

符号约定:**ψ > 0 = 推向冲突,ψ < 0 = 推离冲突**(归因目标是 conflict
logit)。

### 6.6 可视化

```bash
python -m model_baseline.attribution.visualize outputs/attribution/run_001/event_1042.json
```

每个事件产出三张 PNG(存于 JSON 同目录,红 = 推向冲突,蓝 = 推离):

1. **agent × 时间段热图**(symmetric)——证据分布在哪;
2. **风险揭示时间线**(chrono)——ψ_τ 柱 + 从 v(∅) 出发的累计曲线,
   哪一秒升险、哪一秒化险;
3. **通道分解条形图**——位置/航向/速度各占多少。

### 6.7 可选:段级精确 Shapley(案例剖析)

对少数重点事件,可对单个 agent 的时间段做**精确**(非 MC)Shapley:
`attribution.yaml` 里 `exact.enabled: true`、`exact.agents: [0]`(0 = ego)。
计算量 2^段数 次 forward(8 段 = 256 次,很快;上限 16 段)。结果写进
JSON 的 `exact` 字段,可用来校验 MC 估计。

---

## 附录:`baseline_binary` 变体

`model.name: baseline_binary` 是 ① 的**数据格式变体**(同一个 LSTM 网络,
改从 `_data.bin`/`_index.csv` 读数据,适合 memmap 打开慢的环境),不算第四个
模型。要跑的话:复制 `configs/model_baseline.yaml`,把 `model.name` 改为
`baseline_binary`,并先完成 2.4 的二进制缓存。
