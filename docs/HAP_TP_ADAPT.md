# GripReach HAP 离线适配手册（旧版 HAP → 新 TP 模型）

> **读者**：你手里只有**旧版 HAP 工程**，看不到训练仓 / 新代码 diff。  
> **目标**：只靠本文 + 对方发来的 `tp_model.json`，把工程改到能跑新模型。  
> **本包模型（2026-09）**：**截长边 1021 维，不做窗内 2–98 正则**。JSON 里必须是 `"robust_p2_p98": false`。  
> **一句话**：必须 **换模型文件 + 改特征提取代码**；只换 JSON 不够。若代码里仍做 p2–p98，维数对了预测也会全错。

---

## 0. 你需要向对方索取的交付物

向模型侧同事确认拿到下面 **2 样**（缺一不可）：

| # | 文件 | 用途 |
|---|------|------|
| 1 | `tp_model.json`（新） | 覆盖 HAP 的 `entry/src/main/resources/rawfile/tp_model.json` |
| 2 | 本文 `HAP_TP_ADAPT.md` | 特征算法与完整可粘贴代码 |

可选：`imu_model.json`（本次 TP 改动**不必**换 IMU，旧 IMU 可继续用）。

收到 `tp_model.json` 后，用任意文本编辑器打开文件**开头**，必须满足：

```text
"feature_dim": 1021
"tp_edge_only": true
"robust_p2_p98": false
```

- 若 `"feature_dim": 1127`：旧全量模型，不要用。  
- 若 `"robust_p2_p98": true`：这是带 2–98 正则的另一版，**不能**和本文代码混用。

---

## 1. 新旧差异（为什么旧 HAP 会挂）

| | 旧（你现在的 HAP） | 本包要对齐的 |
|--|--|--|
| 特征维 | **1127**（或已改 1021 但带 p2–p98） | **1021** |
| 网格用法 | 全屏 34×21 + LineData | **只留左右长边各 3 列**，LineData **不用** |
| 归一化 | 可能做了窗内 2–98 | **不要做 p2–p98**；截边后直接算统计特征 |
| 输入帧 | 整帧差分 | **不变**：仍喂整帧 `34×21 + 55`，裁边在特征里做 |

典型报错：

```text
feature dim 1021 != 1127
```

含义：代码已按 1021 算 / 或模型是 1021，另一边还是 1127。两边必须同为 1021，且特征算法一致。

**1127 ≠ “更完整的大模型”。** 它只是「全网格特征」；1021 是「截长边后的同任务模型」。

---

## 2. 更新步骤总览（按顺序做）

1. 备份旧 `tp_model.json`、旧 `TpFeatures.ets`（可选）。
2. 用新文件覆盖 `rawfile/tp_model.json`，确认 `feature_dim=1021`。
3. **整文件替换** `TpFeatures.ets` 为本文第 4 节代码（或按第 4 节逐项改）。
4. 检查 `TpPredictor.ets`：滑窗 11 帧、窗内 cumsum、仍调 `fromAbsoluteWindow`（见第 5 节）。
5. DevEco：**Clean Project** → Rebuild → 全量安装（不要只热更资源）。
6. 打日志确认 `feats.length === 1021` 且等于 `model.feature_dim`。
7. 接真机 TP：`pushDiffFrame(整帧一阶差分)`（见第 6 节）。

---

## 3. 算法规格（实现必须逐字对齐）

### 3.1 每帧输入布局（与采集一致，不要先裁边）

```text
flatDiff 长度建议 = 714 + 55 = 769
  [0 .. 713]   : 34 行 × 21 列，行优先 (row 0..33, col 0..20)
  [714 .. 768] : LineData（特征阶段直接丢弃，可传可不传；传了也不进特征）
```

数值含义：**一阶差分**，不是已积分的绝对值。

### 3.2 滑窗

- 目标时长约 **150ms**。
- TP 约 13–14ms/帧 → 取最近 **11 帧**（`nFramesWindow = 11`）。
- 不足 11 帧：**不预测**。

### 3.3 窗内积分（cumsum）

仅对**当前 11 帧窗口**做累加，窗起点相对电平为 0：

```text
abs[0] = diff[0]
abs[f] = abs[f-1] + diff[f]   (f = 1..10)
```

不要用「开机以来无穷积分」当特征输入。

### 3.4 截长边（每帧 714 → 204）

对每帧 flat 的前 714 个数：

- 保留 `col = 0,1,2`（左）
- 保留 `col = 18,19,20`（右）
- 每行顺序：先左 3 再右 3，行优先  
→ 长度 `34 * 6 = 204`。

内圈与 LineData **全部丢弃**。

### 3.5 不要做窗内 2–98

本包模型在训练时**没有**对窗内积分/差分做 p2–p98 → [0,1]。

- **不要**调用 `robustUnitInPlace` / 百分位拉伸。  
- 截长边之后，用原始相对积分值和差分直接拼 3.6 的统计量。  
- JSON 里的 `scaler_mean` / `scaler_scale` 仍由 `RandomForestInfer` 使用，那是特征拼完之后的 StandardScaler，**不是** p2–p98。

### 3.6 拼 1021 维（顺序不可改）

设 `edgeAbs[f][i]`、`edgeDiff[f][i]` 为截边后的窗内积分 / 差分（**未**做 p2–p98）；`F=11`，`E=204`。

| 顺序 | 内容 | 维数 |
|------|------|------|
| 1 | `edgeAbs[F-1][i] - mean_t[i]`（时间去均值后的末帧） | 204 |
| 2 | `mean_t[i] - mean(mean_t)` | 204 |
| 3 | `std_t[i]`（对 edgeAbs，`sqrt(sum((x-mean)^2)/F)`） | 204 |
| 4 | `edgeDiff[F-1][i]` | 204 |
| 5 | `mean(|edgeDiff|)` 标量 | 1 |
| 6 | `edgeAbs[F-1][i] - edgeAbs[0][i]` | 204 |
| | **合计** | **1021** |

然后交给已有的 `RandomForestInfer.predict(feats)`（内部再用 JSON 的 `scaler_mean` / `scaler_scale`）。

标签字符串（勿改）：`左to右` / `右to左` / `others`。

---

## 4. 完整替换：`TpFeatures.ets`

路径（相对你的 HAP 工程）：

```text
entry/src/main/ets/model/TpFeatures.ets
```

**建议整文件覆盖**为下面内容（与训练侧对齐，可直接编译）：

```ts
/** TP window features — long-edge 3 layers, NO p2–p98, 1021 dims */

export class TpFeatureExtractor {
  static readonly GRID_ROWS: number = 34;
  static readonly GRID_COLS: number = 21;
  static readonly EDGE_LAYERS: number = 3;

  static extractFromDiffWindow(framesDiff: number[][], lineLen: number = 55): number[] | null {
    const F = framesDiff.length;
    if (F < 3) {
      return null;
    }
    const cell = TpFeatureExtractor.GRID_ROWS * TpFeatureExtractor.GRID_COLS;
    const gridAbs: number[][] = [];
    const gridDiff: number[][] = [];
    let gPrev: number[] = new Array(cell).fill(0);
    for (let f = 0; f < F; f++) {
      const gd = framesDiff[f].slice(0, cell);
      const ga: number[] = new Array(cell);
      for (let i = 0; i < cell; i++) {
        ga[i] = gPrev[i] + gd[i];
      }
      gPrev = ga;
      gridAbs.push(ga);
      gridDiff.push(gd);
    }
    const dummy: number[][] = [];
    return TpFeatureExtractor.fromAbsoluteWindow(gridAbs, dummy, gridDiff, dummy);
  }

  static fromAbsoluteWindow(
    gridAbs: number[][],
    _lineAbs: number[][],
    gridDiff: number[][],
    _lineDiff: number[][]
  ): number[] {
    const F = gridAbs.length;
    const edgeAbs: number[][] = [];
    const edgeDiff: number[][] = [];
    for (let f = 0; f < F; f++) {
      edgeAbs.push(TpFeatureExtractor.longEdge(gridAbs[f]));
      edgeDiff.push(TpFeatureExtractor.longEdge(gridDiff[f]));
    }
    const E = edgeAbs[0].length;

    const gMean: number[] = new Array(E).fill(0);
    for (let f = 0; f < F; f++) {
      for (let i = 0; i < E; i++) {
        gMean[i] += edgeAbs[f][i];
      }
    }
    for (let i = 0; i < E; i++) {
      gMean[i] /= F;
    }

    const feats: number[] = [];
    for (let i = 0; i < E; i++) {
      feats.push(edgeAbs[F - 1][i] - gMean[i]);
    }
    let globalMean = 0;
    for (let i = 0; i < E; i++) {
      globalMean += gMean[i];
    }
    globalMean /= E;
    for (let i = 0; i < E; i++) {
      feats.push(gMean[i] - globalMean);
    }
    const gStd: number[] = new Array(E).fill(0);
    for (let f = 0; f < F; f++) {
      for (let i = 0; i < E; i++) {
        const d = edgeAbs[f][i] - gMean[i];
        gStd[i] += d * d;
      }
    }
    for (let i = 0; i < E; i++) {
      feats.push(Math.sqrt(gStd[i] / F));
    }
    for (let i = 0; i < E; i++) {
      feats.push(edgeDiff[F - 1][i]);
    }
    let absG = 0;
    let cntG = 0;
    for (let f = 0; f < F; f++) {
      for (let i = 0; i < E; i++) {
        absG += Math.abs(edgeDiff[f][i]);
        cntG++;
      }
    }
    feats.push(absG / Math.max(1, cntG));
    for (let i = 0; i < E; i++) {
      feats.push(edgeAbs[F - 1][i] - edgeAbs[0][i]);
    }
    return feats;
  }

  /** Left 3 cols then right 3 cols, row-major. */
  static longEdge(flat: number[]): number[] {
    const rows = TpFeatureExtractor.GRID_ROWS;
    const cols = TpFeatureExtractor.GRID_COLS;
    const layers = TpFeatureExtractor.EDGE_LAYERS;
    const out: number[] = new Array(rows * layers * 2);
    let k = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < layers; c++) {
        out[k++] = flat[r * cols + c];
      }
      for (let c = cols - layers; c < cols; c++) {
        out[k++] = flat[r * cols + c];
      }
    }
    return out;
  }
}
```

### 若你不想整文件替换，旧代码至少要改到

1. 删掉「全网格 + LineData → 1127」的拼特征逻辑。  
2. 增加 `EDGE_LAYERS=3`、`longEdge`。  
3. `fromAbsoluteWindow`：**只** `longEdge`，**不要** p2–p98，再按 3.6 顺序拼 1021 维。  
4. 若旧代码里有 `robustUnitInPlace`，必须删掉调用。  
5. 对外方法签名尽量保持 `fromAbsoluteWindow(gridAbs, lineAbs, gridDiff, lineDiff)`，这样 `TpPredictor` 可少改；`line*` 参数可忽略。

---

## 5. 检查 / 微调：`TpPredictor.ets`

路径：`entry/src/main/ets/model/TpPredictor.ets`

旧版若已有 `pushDiffFrame` + 窗内 cumsum，通常**只需确认**：

| 项 | 要求 |
|----|------|
| `nFramesWindow` | `11` |
| `pushDiffFrame` | 缓存整帧差分，长度约 `714+55` |
| 预测前 | 对最近 11 帧做窗内 cumsum，得到 `gridAbs` / `gridDiff`（line 可算可不算） |
| 调特征 | `TpFeatureExtractor.fromAbsoluteWindow(...)` |
| 推理 | `this.infer.predict(feats)` |

参考逻辑（不必一字不差，语义要对）：

```ts
// 伪代码
const window = frames.slice(-11);
let g = zeros(714);
const gridAbs = [], gridDiff = [];
for (raw of window) {
  const gd = raw.slice(0, 714);
  const ga = g.zipAdd(gd);  // 逐元素累加
  g = ga;
  gridAbs.push(ga);
  gridDiff.push(gd);
}
const feats = TpFeatureExtractor.fromAbsoluteWindow(gridAbs, [], gridDiff, []);
// feats.length 必须是 1021
const result = this.infer.predict(feats);
```

`RandomForestInfer.ets`：若已有 `features.length !== model.feature_dim` 检查，**不用改**；换新 JSON 后自然要求 1021。

---

## 6. 替换模型文件

1. 把对方发来的新 `tp_model.json` 放到：

```text
entry/src/main/resources/rawfile/tp_model.json
```

2. 自检（本机有 Python 时）：

```bash
python -c "import json; d=json.load(open(r'entry/src/main/resources/rawfile/tp_model.json',encoding='utf-8')); print(d['feature_dim'], d.get('tp_edge_only'), d.get('robust_p2_p98'), d['n_estimators'])"
```

期望输出类似：

```text
1021 True False 400
```

无 Python：直接搜 `"feature_dim"` 必须是 `1021`，且 `"robust_p2_p98": false`。

3. DevEco：**Clean Project** → Build → 安装。确认 rawfile 进包（曾出现「编译过但包内仍是旧 json」）。

4. 运行时打一次日志：

```ts
console.info(`tp feats=${feats.length} model_dim=${this.infer /* 或 model.feature_dim */}`);
```

两者都必须是 **1021**。

---

## 7. 接真机 TP（你侧主开发）

公开 HarmonyOS API **通常拿不到**训练用的电容差分矩阵。请把厂商 / 系统回调接到：

```ts
tpPredictor.pushDiffFrame(flatDiff);
```

要求：

- `flatDiff` = **一阶差分**，布局见 3.1  
- **不要**在驱动里先裁成只有长边再送入（会与 `longEdge` 双重裁切）  
- 行列约定与采集一致：34×21 行优先  
- UI：`左to右` → 右半屏绿约 3s；`右to左` → 左半屏绿；`others` → 黑  
- 演示按钮（若有 `injectDemoPulse`）只测 UI，**不算**真 TP 验收

---

## 8. IMU

本次可不改。`imu_model.json` 仍是约 78 维；与 TP 模态独立。换 TP 特征不必动 IMU 推理。

---

## 9. 常见问题

| 现象 | 处理 |
|------|------|
| `feature dim 1021 != 1127` | 模型或特征仍一边旧：确认 rawfile 是 1021，且 `TpFeatures` 已按本文替换 |
| 维数都 1021，预测全 `others` / 乱跳 | 代码里仍做了 p2–p98，或左右列取反，或送的不是差分 |
| Clean 后行为仍像旧包 | 确认安装的是新 HAP；核对设备上资源是否更新 |
| 自己拼 1127 维喂新模型 | 禁止；必须走本文特征管线 |
| 采集端先裁边再 `pushDiffFrame` | 禁止；送整帧 714（+55） |

---

## 10. 验收检查单（做完打勾）

- [ ] 收到的 `tp_model.json`：`feature_dim=1021`，`tp_edge_only=true`，**`robust_p2_p98=false`**
- [ ] 已覆盖到 `rawfile/tp_model.json`
- [ ] `TpFeatures.ets` 已按第 4 节更新（`longEdge`，**没有** p2–p98）
- [ ] `TpPredictor`：11 帧窗 + 窗内 cumsum + 调 `fromAbsoluteWindow`
- [ ] Clean + 全量重装
- [ ] 运行时 `feats.length === 1021`
- [ ] 真机 `pushDiffFrame` 已接（演示按钮不算）

---

## 11. 给对方的最短回执模板

更新完成后可回：

> rawfile `feature_dim=1021` 且 `robust_p2_p98=false` 已确认；`TpFeatures` 已改为长边 3 层且**没有** p2–p98；运行时 feats=1021；Clean 重装完成。TP 回调已/未接到 `pushDiffFrame`。
