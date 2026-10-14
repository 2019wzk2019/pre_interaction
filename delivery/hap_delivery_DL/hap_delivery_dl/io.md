# GripReachPredict HAP 适配说明（DL 全屏单帧 TP）

> **给 HAP 同事**：只看本文 + 同目录 `tp_model.onnx` + `tp_model_meta.json` 即可完成解析、预处理、推理、亮灯。  
> **不要用旧包** `hap_delivery_500ms_RF`（500ms 滑窗、截长边 1021 维、随机森林 JSON）。本包 **不能** 只换文件、沿用旧 `TpFeatures.ets`。  
> 任务仍是三分类 `左to右` / `右to左` / `others`。  
> - `左to右` = 手从屏幕**左**边缘伸入（够右下）→ 建议亮**右**半屏  
> - `右to左` = 手从屏幕**右**边缘伸入（够左下）→ 建议亮**左**半屏

本目录文件：

| 文件 | 必须？ | 用途 |
|------|--------|------|
| `tp_model.onnx` | 是 | ConvNet 权重（约 726KB，单文件，无 sidecar） |
| `tp_model_meta.json` | 是 | 标签、输入形状、归一化约定 |
| 本文 `io.md` | 是 | 集成说明 |

把 onnx 放到（路径可按工程调整，但 rawfile 必须能读到）：

```text
entry/src/main/resources/rawfile/tp_model.onnx
entry/src/main/resources/rawfile/tp_model_meta.json
```

DevEco **Clean → 全量编译安装**。不要只热更资源。本次 **不必** 换 IMU 模型。

实验室（`data/tp/train` 训练 / `data/tp/test` 留出）：holdout acc ≈ 99.96%；点击类仍可能偶发误报，**建议保留门控，不要裸 argmax 亮灯**。

---

## 0. 和旧版 RF（`hap_delivery_500ms_RF`）IO 有没有区别？

**采集端原始回调可以不变**：仍是厂商每帧送来的 **一阶差分**，34×21 行优先，LineData 可带可不带。

**特征 / 滑窗 / 模型入口全部变了**，旧 HAP 只换 onnx、不改预处理会全错。

| 项 | 旧 RF `hap_delivery_500ms` | 本包 DL |
|----|----------------------------|---------|
| 设备原始输入 | 整帧一阶差分，建议 714+55 | **相同** |
| 采集端是否裁边 | 不要裁，送整帧 | **相同**（本包特征里也不裁边） |
| 滑窗 | **500ms**，约 42 帧（`round(500/pollMs)`） | **单帧**：每来一帧就能推一次 |
| 网格用法 | **只留长边各 3 列**（204 格） | **全屏 34×21 = 714** |
| 手工特征 | 六块统计 → **1021 维** | **无**；714 个数直接进网络 |
| cumsum / 去均值 / p2–p98 | 无积分；有窗内 demean 统计 | **都不要** |
| 标准化 | 训练好的 `scaler_mean/scale`（1021 维） | **这一帧 714 个数除以自身标准差** |
| 模型文件 | `tp_model.json` 随机森林 | `tp_model.onnx` ConvNet |
| 网络输入张量 | 无（特征是 1D） | `tp_grid`，形状 **`(1, 1, 34, 21)`** NCHW |
| 网络输出 | 三类概率（树平均，已归一） | `logits` 长度 3，**要 softmax** 才是概率 |
| 标签顺序 | `左to右, 右to左, others` | **相同** |
| 亮灯语义 | 左to右→右半屏；右to左→左半屏 | **相同** |

一句话：**传感器 IO 兼容；HAP 特征与推理 IO 不兼容。**

---

## 1. 装包前自检

打开 `tp_model_meta.json`，必须同时满足：

```text
"backend": "onnx_cnn"
"labels": ["左to右", "右to左", "others"]
"n_frames": 1
"single_frame": true
"window_ms": 0
"input_name": "tp_grid"
"output_name": "logits"
"input_shape": [1, 1, 34, 21]
"normalize": "per_sample_std"
"tp_edge_only": false
"feature_dim": 714
"uses_cumsum_absolute": false
"robust_p2_p98": false
```

有 Python 时：

```bash
python -c "import json,onnx; m=json.load(open('tp_model_meta.json',encoding='utf-8')); print(m['n_frames'], m['input_shape'], m['normalize'], m['tp_edge_only']); n=onnx.load('tp_model.onnx'); print([(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in n.graph.input])"
```

期望：`1 [1, 1, 34, 21] per_sample_std False`，输入名 `tp_grid`。

任一不符：**不要用**，让模型侧重发。

---

## 2. 设备侧原始输入（与旧 RF 相同）

每帧喂 **一阶差分**，不是绝对电容、不是已积分值。

建议长度 `714 + 55 = 769`：

```text
flatDiff[0 .. 713]   : 34 行 × 21 列，行优先
                       index = r * 21 + c，r=0..33，c=0..20
flatDiff[714 .. 768] : LineData，本包丢弃（可传可不传）
```

**采集端不要先裁边。** 本包用全屏 714，裁了维数对不上。

---

## 3. 预处理：1 帧差分 → `(1,1,34,21)`

**禁止：**

- 攒 500ms / 42 帧再推（可以每帧都推）
- 截长边 3 列
- 算 1021 维统计特征
- 窗内/开机 cumsum
- p2–p98
- 用旧 json 里的 `scaler_mean/scale`
- 行列转置、左右列对调

**只做：** 取前 714 个数 → 按行填进 34×21 → 整帧除以标准差。

```ts
function stdNormalizeAndPack(flatDiff: number[]): Float32Array {
  const rows = 34, cols = 21, n = rows * cols; // 714
  const src = flatDiff.slice(0, n);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += src[i];
  mean /= n;
  let varSum = 0;
  for (let i = 0; i < n; i++) {
    const d = src[i] - mean;
    varSum += d * d;
  }
  // 与训练一致：除以总体标准差 std(x)（含均值，numpy 默认）
  let std = Math.sqrt(varSum / n);
  if (std < 1e-6) std = 1;
  const out = new Float32Array(n); // NCHW: 1*1*34*21
  for (let i = 0; i < n; i++) out[i] = src[i] / std;
  return out;
}
```

训练代码是 `x / std(x)`（没有先减均值再除，std 本身按有均值的总体标准差算）。上面先算 std 再 `src[i]/std`，**不要**写成 `(src[i]-mean)/std`。

若 ArkTS 不便先算 mean：等价于 numpy `np.std(x)`（`ddof=0`）后逐元素相除。

环形缓冲：单帧模型 **不必** 再维持 42 帧窗口。每帧 `pushDiffFrame` 即可推理。流启动后前几帧基线不稳，可先不预测。

```ts
pushDiffFrame(flatDiff: number[]): void {
  const tensor = stdNormalizeAndPack(flatDiff); // 长度 714，layout NCHW 1x1x34x21
  predict(tensor);
}
```

---

## 4. 推理：ONNX → 三个 logit → 概率

输入：

```text
name:  tp_grid
dtype: float32
shape: [1, 1, 34, 21]
内存:  行优先，与 flat 714 同序（第 r 行紧挨着放 21 列）
```

输出：

```text
name:  logits
dtype: float32
shape: [1, 3]
含义:  未归一化分数，不是概率
下标:  0=左to右, 1=右to左, 2=others
```

```text
1) 跑 ONNX，得到 logits[3]
2) m = max(logits)
   p[c] = exp(logits[c]-m) / sum_k exp(logits[k]-m)
3) 裸 label = labels[argmax(p)]
   proba["左to右"]=p[0]，proba["右to左"]=p[1]，proba["others"]=p[2]
```

**不要**把 logit 当概率用，也 **不要** 再走 `RandomForestInfer`。

Harmony 侧用你们现有的 ONNX Runtime / NNRt 加载 `tp_model.onnx`。若接口要求 NHWC，必须先转成文档中的 NCHW，否则空间会错。

---

## 5. 输出怎么用（建议仍做门控）

标签语义与旧 RF **相同**。DL 实验室 acc 很高，真机仍建议阈值 + 投票，避免点击闪一下。

可沿用旧包附录 C 的 `DecisionGate`（`L2R_MIN=0.80`，`R2L_MIN=0.65`，`GAP=0.20`，5 窗 3 票）。注意门控吃的是 **softmax 后的 pL/pR/pO**。

| 门控后的标签 | 含义 | 建议 |
|--------------|------|------|
| `左to右` | 左手够右下 | 右半屏绿 |
| `右to左` | 右手够左下 | 左半屏绿 |
| `others` | 其它 | 黑 |

日志必须打出 `pL / pR / pO` 和 `argmax`。

---

## 6. 集成步骤

1. 放入 `tp_model.onnx` + `tp_model_meta.json`，做第 0 节自检。  
2. **删掉/停用** 旧 500ms 滑窗和 `TpFeatures` 1021 维。  
3. 每帧：714 差分 → std 归一化 → `(1,1,34,21)` → ONNX。  
4. logits 做 softmax；亮灯按第 5 节。  
5. Clean 全量安装。  
6. 真机 `pushDiffFrame` 接厂商**整帧**差分回调。

---

## 7. 常见错误

| 现象 | 处理 |
|------|------|
| 仍算 1021 维 / 调 RandomForestInfer | 本包不是 RF |
| 仍攒 42 帧 | 单帧模型，多帧不会自动对 |
| 只送长边 6 列 | 必须 34×21 全屏 |
| 还在 cumsum | 直接喂差分 |
| 用 `(x-mean)/std` | 训练是 `x/std(x)`，不要减均值 |
| 把 logits 当概率做门控 | 先 softmax |
| NHWC 当 NCHW | 布局错，预测会糊 |
| 三类≈0.33 | 查归一化、行列、是否喂了绝对值 |

---

## 8. 验收

- [ ] meta：`n_frames=1`，`input_shape=[1,1,34,21]`，`tp_edge_only=false`，`normalize=per_sample_std`  
- [ ] 无 500ms 窗、无 1021 特征、无截边、无积分  
- [ ] 运行时输入 714 float，输出 3 logits  
- [ ] 日志有 softmax 后的 `pL/pR/pO`  
- [ ] Clean 重装  
- [ ] 静置/握持为 others；左伸入亮右半屏；右伸入亮左半屏  

回执：

> json/onnx 单帧全屏 / per_sample_std 已确认；输入 1x1x34x21；softmax 后静置 pO≈___；左伸 pL≈___；右伸 pR≈___；握持误报：（有/无）。

---

## 附录 A — 预处理（可粘贴）

```ts
export class TpFramePrep {
  static readonly GRID_ROWS: number = 34;
  static readonly GRID_COLS: number = 21;
  static readonly EPS: number = 1e-6;

  /** flatDiff 至少 714；返回 NCHW float32 长度 714。 */
  static pack(flatDiff: number[]): Float32Array {
    const n = TpFramePrep.GRID_ROWS * TpFramePrep.GRID_COLS;
    if (flatDiff.length < n) {
      throw new Error(`flatDiff ${flatDiff.length} < ${n}`);
    }
    let sum = 0;
    let sum2 = 0;
    for (let i = 0; i < n; i++) {
      const v = flatDiff[i];
      sum += v;
      sum2 += v * v;
    }
    const mean = sum / n;
    let std = Math.sqrt(Math.max(0, sum2 / n - mean * mean));
    if (std < TpFramePrep.EPS) {
      std = 1;
    }
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = flatDiff[i] / std;
    }
    return out;
  }

  static softmax(logits: number[]): number[] {
    let m = logits[0];
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > m) m = logits[i];
    }
    const ex: number[] = [];
    let s = 0;
    for (let i = 0; i < logits.length; i++) {
      const e = Math.exp(logits[i] - m);
      ex.push(e);
      s += e;
    }
    return ex.map((e) => e / s);
  }
}
```

预测时：`pack(frame.slice(0,714))` 作为 `tp_grid`；`softmax(logits)` 得到三类概率。

---

## 附录 B — 门控（可与旧 RF 包相同）

```ts
export type GateLabel = '左to右' | '右to左' | 'others';

export class DecisionGate {
  l2rMin: number = 0.80;
  r2lMin: number = 0.65;
  gapOthers: number = 0.20;
  voteWindow: number = 5;
  needHits: number = 3;
  private recent: GateLabel[] = [];

  reset(): void { this.recent = []; }

  step(pL: number, pR: number, pO: number): GateLabel {
    const hitL = (pL >= this.l2rMin) && (pL - pO >= this.gapOthers);
    const hitR = (pR >= this.r2lMin) && (pR - pO >= this.gapOthers);
    let cand: GateLabel = 'others';
    if (hitL && hitR) {
      cand = (pL >= pR) ? '左to右' : '右to左';
    } else if (hitL) {
      cand = '左to右';
    } else if (hitR) {
      cand = '右to左';
    }
    this.recent.push(cand);
    if (this.recent.length > this.voteWindow) {
      this.recent.shift();
    }
    let nL = 0, nR = 0;
    for (let i = 0; i < this.recent.length; i++) {
      if (this.recent[i] === '左to右') nL++;
      if (this.recent[i] === '右to左') nR++;
    }
    if (nL >= this.needHits && nL >= nR) return '左to右';
    if (nR >= this.needHits && nR > nL) return '右to左';
    return 'others';
  }
}
```
