# GripReachPredict HAP 适配说明（IMU 100ms DL v3）

> **给 HAP 同事**：只看本文 + 同目录 `imu_model.onnx` + `imu_model_meta.json`。  
> 这是 **只用 IMU** 的三分类，**不是** TP 电容模型，也 **不是** 旧 RF `imu_model.json`。  
> **不要** 把本包接到 `TpFeatures` / 电容差分上。  
> **IO 与 `hap_delivery_IMU_100ms` / v2 完全相同**：只换 `imu_model.onnx` + `imu_model_meta.json` 即可，预处理不用改。  
> 任务：`左to右` / `右to左` / `others`  
> - `左to右` = 手从屏幕**左**边伸入（够右下）→ 建议亮**右**半屏  
> - `右to左` = 手从屏幕**右**边伸入（够左下）→ 建议亮**左**半屏  
> - `others` = 握持 / 静置 / 点击 / 伸手回程 等

本目录：

| 文件 | 必须？ | 用途 |
|------|--------|------|
| `imu_model.onnx` | 是 | 1D ConvNet，约 122KB，单文件 |
| `imu_model_meta.json` | 是 | 窗长、通道顺序、形状 |
| 本文 `io.md` | 是 | 集成说明 |

建议放到：

```text
entry/src/main/resources/rawfile/imu_model.onnx
entry/src/main/resources/rawfile/imu_model_meta.json
```

DevEco **Clean → 全量编译安装**。本次 **不必** 换 TP 模型；IMU 与 TP 可并存，但推理管线要分开。

实验室（`data/imu_old` onset 切滑 + 更新后的 `data/imu_new` 前后两段，含新增 OTHERS）：窗 acc ≈ **93.1%**，片段多数票 **21/21**。左右伸入 recall 很高；回程窗仍可能被当成滑动，**禁止裸 argmax 亮灯**。

---

## 0. 和旧 IMU RF / TP 包的区别

| 项 | 旧 IMU RF（若有） | 本包 |
|----|-------------------|------|
| 输入 | 150ms 手工统计（约 78 维） | **100ms 原始波形** |
| 模型 | `imu_model.json` 随机森林 | `imu_model.onnx` |
| 张量 | 无 | `imu_win` **`(1, 6, 10)`** |
| 输出 | 三类概率 | 三类 **logits**，先 **softmax** |
| 滑窗 | 150ms / 50ms | **100ms 窗，建议每 10ms 推一次** |
| 标准化 | scaler_mean/scale | **这一窗 60 个数除以自身 std** |

和 TP DL 包（`hap_delivery_DL`）**完全不是一个输入**：那边是 34×21 电容；这边是 acc+gyro。

训练时 L2R/R2L 用 gyro 切掉了开头的静握持，**真机推理不要做 onset 切割**，只滑最近 100ms 即可。

---

## 1. 装包前自检

`imu_model_meta.json` 必须满足：

```text
"modality": "imu"
"backend": "onnx_cnn"
"window_ms": 100
"rate_hz": 100
"n_steps": 10
"n_channels": 6
"channel_order": ["ax","ay","az","gx","gy","gz"]
"input_name": "imu_win"
"output_name": "logits"
"input_shape": [1, 6, 10]
"normalize": "per_sample_std"
"labels": ["左to右", "右to左", "others"]
```

```bash
python -c "import json,onnx; m=json.load(open('imu_model_meta.json',encoding='utf-8')); print(m['window_ms'], m['input_shape'], m['channel_order']); n=onnx.load('imu_model.onnx'); print([(i.name,[d.dim_value for d in i.type.tensor_type.shape.dim]) for i in n.graph.input])"
```

期望：`100 [1, 6, 10] ['ax','ay','az','gx','gy','gz']`，输入名 `imu_win`。

---

## 2. 设备侧原始输入

系统传感器：

```text
acc  : 加速度，单位 m/s²，三轴 x,y,z（设备坐标系，与系统 Sensor.ACCELEROMETER 一致）
gyro : 角速度，单位 rad/s，三轴 x,y,z（Sensor.GYROSCOPE）
时间 : 用传感器时间戳（ns 或 ms 均可），两路都要有
```

**不要**用线性加速度（去重力）替代 acc。训练用的是带重力的 `acc`。  
采样率真机常见 100–500Hz，预处理会重采样到 **100Hz**，不必在驱动里改成 100Hz。

环形缓冲建议各存最近 **≥150ms** 的 acc / gyro。不足 100ms **不推理**。

---

## 3. 预处理：100ms 缓冲 → `(1, 6, 10)`

对当前时刻 `tEnd`：

1. 取 `[tEnd-100ms, tEnd]`。acc、gyro 都要覆盖这段，否则跳过。  
2. 生成 10 个均匀时刻：

```text
rate = 100
n = 10
t[i] = tEnd - 100ms + i * (100ms / 9)     // i=0..9，含两端
```

训练实现是 `linspace(t0, t1, 10)`，即两端都 inclusive、中间 8 点等分。

3. 对每个轴把原始样本 **线性插值** 到这 10 个时刻。  
4. 拼通道，**顺序不能改**：

```text
row i:  [ax, ay, az, gx, gy, gz]     // i = 0..9 时间从旧到新
```

内存布局 **(N,C,T)**：先 10 个 ax，再 10 个 ay，… 最后 10 个 gz。  
也就是 `data[c * 10 + t] = channel[c][t]`。

5. **整窗除以标准差**（60 个数一起算，**不要减均值**）：

```text
std = sqrt( mean(x^2) - mean(x)^2 )     // 总体标准差，ddof=0
if std < 1e-6: std = 1
x[i] = x[i] / std
```

**禁止：** 只喂 gyro、通道对调、NHWC、用 150ms 旧特征、对 IMU 做 TP 的截边/cumsum。

---

## 4. 推理

```text
输入  name=imu_win  float32  shape [1, 6, 10]
输出  name=logits   float32  shape [1, 3]
      0=左to右  1=右to左  2=others   （未归一，先 softmax）
```

```text
m = max(logits)
p[c] = exp(logits[c]-m) / sum exp(logits[k]-m)
裸 label = labels[argmax(p)]
```

建议每 **10ms**（或每 1～2 个 sensor 事件）推一次。不要等 100ms 才跳一步。

---

## 5. 亮灯门控（必做）

others 窗约有 10% 会被当成滑动。裸 `argmax` 会闪。

可沿用 TP 那套（输入必须是 **softmax 后的概率**）：

```text
hitL = (pL >= 0.80) && (pL - pO >= 0.20)
hitR = (pR >= 0.65) && (pR - pO >= 0.20)
最近 5 次 cand 同一侧 ≥ 3 票才亮
```

| 门控标签 | 建议 |
|----------|------|
| `左to右` | 右半屏绿 |
| `右to左` | 左半屏绿 |
| `others` | 黑 |

日志打 `pL / pR / pO` 和 `argmax`。

---

## 6. 集成步骤

1. rawfile 放入 onnx + meta，做第 0 节自检。  
2. 订阅 acc + gyro，环形缓冲 ≥150ms。  
3. 每 10ms：插值 100ms → 6×10 → `/std` → ONNX → softmax。  
4. 门控亮灯。  
5. Clean 全量安装。

---

## 7. 常见错误

| 现象 | 处理 |
|------|------|
| 形状不是 1×6×10 | C 和 T 反了：应是 6 通道 × 10 步 |
| 三类≈0.33 | 忘了 /std，或通道顺序错，或用了线性加速度 |
| 握持乱亮 | 未做第 5 节门控 |
| 接了电容差分 | 本包只吃 IMU |
| 仍跑 RandomForestInfer | 本包是 ONNX |
| 把 logits 当概率 | 先 softmax |
| 等满 100ms 才输出一帧 | 应滑动，stride≈10ms |

---

## 8. 验收

- [ ] meta：`window_ms=100`，`input_shape=[1,6,10]`，通道 ax..gz  
- [ ] 真机 acc 是带重力的加速度  
- [ ] 不足 100ms 不推  
- [ ] 日志有 softmax 后 pL/pR/pO  
- [ ] 门控 + 5 窗 3 票  
- [ ] 静置/握持黑；左伸入亮右；右伸入亮左  

回执：

> IMU 100ms v3 ONNX / 1x6x10 / per_sample_std 已确认。静置 pO≈___；左伸 pL≈___；右伸 pR≈___；握持误报：（有/无）。

---

## 附录 A — 预处理（可粘贴）

```ts
export class ImuWinPrep {
  static readonly WINDOW_MS: number = 100;
  static readonly N_STEPS: number = 10;
  static readonly N_CH: number = 6;
  static readonly EPS: number = 1e-6;

  /** acc/gyro: {tMs, x, y, z}[] 已按时间升序。返回 NCHW 长度 60 的 float32，或 null。 */
  static pack(
    acc: Array<{ tMs: number; x: number; y: number; z: number }>,
    gyro: Array<{ tMs: number; x: number; y: number; z: number }>,
    tEndMs: number
  ): Float32Array | null {
    const t0 = tEndMs - ImuWinPrep.WINDOW_MS;
    if (acc.length < 2 || gyro.length < 2) {
      return null;
    }
    if (acc[0].tMs > t0 || acc[acc.length - 1].tMs < tEndMs) {
      return null;
    }
    if (gyro[0].tMs > t0 || gyro[gyro.length - 1].tMs < tEndMs) {
      return null;
    }
    const n = ImuWinPrep.N_STEPS;
    const ch: number[][] = [[], [], [], [], [], []];
    for (let i = 0; i < n; i++) {
      const t = t0 + (ImuWinPrep.WINDOW_MS * i) / (n - 1);
      const a = ImuWinPrep.lerp3(acc, t);
      const g = ImuWinPrep.lerp3(gyro, t);
      ch[0].push(a[0]); ch[1].push(a[1]); ch[2].push(a[2]);
      ch[3].push(g[0]); ch[4].push(g[1]); ch[5].push(g[2]);
    }
    const out = new Float32Array(ImuWinPrep.N_CH * n);
    let k = 0;
    let sum = 0;
    let sum2 = 0;
    for (let c = 0; c < 6; c++) {
      for (let i = 0; i < n; i++) {
        const v = ch[c][i];
        out[k++] = v;
        sum += v;
        sum2 += v * v;
      }
    }
    const mean = sum / out.length;
    let std = Math.sqrt(Math.max(0, sum2 / out.length - mean * mean));
    if (std < ImuWinPrep.EPS) {
      std = 1;
    }
    for (let i = 0; i < out.length; i++) {
      out[i] = out[i] / std;
    }
    return out;
  }

  static lerp3(
    s: Array<{ tMs: number; x: number; y: number; z: number }>,
    t: number
  ): number[] {
    if (t <= s[0].tMs) {
      return [s[0].x, s[0].y, s[0].z];
    }
    const last = s[s.length - 1];
    if (t >= last.tMs) {
      return [last.x, last.y, last.z];
    }
    let lo = 0;
    let hi = s.length - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (s[mid].tMs <= t) lo = mid; else hi = mid;
    }
    const a = s[lo];
    const b = s[hi];
    const u = (t - a.tMs) / Math.max(1e-6, b.tMs - a.tMs);
    return [
      a.x + (b.x - a.x) * u,
      a.y + (b.y - a.y) * u,
      a.z + (b.z - a.z) * u,
    ];
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

`pack(...)` 的 `Float32Array` 直接作为 ONNX 输入 `imu_win`。

---

## 附录 B — 门控

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
