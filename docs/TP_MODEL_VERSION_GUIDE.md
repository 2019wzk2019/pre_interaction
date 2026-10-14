# TP 模型版本核对（给同事）

同事**只有旧版 HAP、看不到本仓库代码**时：把 **[HAP_TP_ADAPT.md](./HAP_TP_ADAPT.md)** + 新 `tp_model.json` 一并发过去。该文档含完整可粘贴的 `TpFeatures.ets`，无需拉训练仓即可改 HAP。

`harmony_app/` 已从本仓库跟踪中移除；同事 `git pull` **不会**用 GitHub 覆盖他改过的 HAP，但若工作区干净、且仍跟踪旧路径，pull 可能**删掉**工作区里的 `harmony_app/`。拉之前请他先备份该目录。

## 结论先说

同事说得对：**不是“没截取长边”的误会**，而是 **HAP 里的 `tp_model.json` 确实还是旧坏模型**。

当前仓库里曾经出现过这种错配：

| 文件 | feature_dim | 含义 |
|------|-------------|------|
| `artifacts/tp_model.joblib`（新） | **1021** | 只保留长边外 3 层 + 窗内 p2–p98 归一化 |
| `.../rawfile/tp_model.json`（旧） | **1127** | 全网格 34×21 + LineData，未截边 |

ArkTS 特征提取 `TpFeatures.ets` 已经按 **长边 3 层** 算特征（输出约 1021 维），但 rawfile 里若仍是 **1127 维** 的 JSON，推理时会直接炸：

```text
feature dim 1021 != 1127
```

所以 TP 预测“全部失败”，不是同事没截取，而是 **代码截了边、模型还是全网格旧版**。

---

## 怎么一眼判断旧/新模型

在工程里打开：

`harmony_app/GripReachPredict/entry/src/main/resources/rawfile/tp_model.json`

看文件开头附近字段（或用下面命令）：

```bash
python -c "import json; d=json.load(open(r'harmony_app/GripReachPredict/entry/src/main/resources/rawfile/tp_model.json',encoding='utf-8')); print('feature_dim=', d['feature_dim']); print('n_estimators=', d['n_estimators']); print('tp_edge_only=', d.get('tp_edge_only')); print('robust_p2_p98=', d.get('robust_p2_p98'))"
```

### 旧坏模型（会挂）

- `feature_dim = 1127`
- 通常没有 `tp_edge_only`
- 可能只有 `tp_grid_rows/tp_grid_cols`

### 新正确模型（应使用）

- `feature_dim = 1021`
- `tp_edge_only = true`
- `robust_p2_p98 = true`（窗内 2–98 分位归一化）
- `n_estimators` 可以是 400（完整版）或更小（手机轻量版），**维数比树数更关键**

MD5 和“上午备份”一致，只说明 **文件没被更新**，不能用来判断算法对不对；要以 `feature_dim` 为准。

---

## 为什么会这样

训练流程改过两轮：

1. **只保留竖屏长边最外 3 列**（左 3 + 右 3），内圈和 LineData 丢掉 → 特征从 1127 → 1021  
2. **每个 150ms 窗内对积分值/差分做 p2–p98 裁剪并缩放到 [0,1]**

`artifacts/tp_model.joblib` 已按新流程重训，但 **忘记/未执行** 把 joblib 导出进 rawfile：

```bash
python scripts/sync_harmony_models.py
```

上次有人同步过一次，同步的是 **当时的 1127 维旧 joblib**；之后 joblib 更新了，JSON 没再同步，于是 HAP 一直装着旧文件。

---

## 正确替换步骤（同事按这个做）

在仓库根目录 `preinteract/`：

```bash
# 1) 确认 joblib 已是新版
python -c "import joblib; b=joblib.load('artifacts/tp_model.joblib'); print(b['feature_dim'], b.get('tp_edge_only'), b.get('robust_p2_p98'))"
# 期望: 1021 True True

# 2) 导出到鸿蒙 rawfile（覆盖 tp_model.json / imu_model.json）
python scripts/sync_harmony_models.py

# 3) 再确认 rawfile
python -c "import json; d=json.load(open(r'harmony_app/GripReachPredict/entry/src/main/resources/rawfile/tp_model.json',encoding='utf-8')); print(d['feature_dim'], d.get('tp_edge_only'), d.get('robust_p2_p98'))"
# 期望: 1021 True True
```

然后：

1. DevEco **Clean Project**  
2. 重新 Build / 安装 HAP（不要只热更新资源；确认 rawfile 进包）  
3. 真机测 TP：若仍无原始电容通道，演示按钮只测 UI；接厂商 `pushDiffFrame` 才是真 TP

若本地没有新 joblib，先重训再同步：

```bash
python scripts/train_grip_models.py --modality tp --window-ms 150
python scripts/sync_harmony_models.py
```

---

## 和特征代码是否对齐

| 组件 | 要求 |
|------|------|
| `TpFeatures.ets` | `EDGE_LAYERS = 3`，长边截取 + `robustUnitInPlace`（p2–p98） |
| `tp_model.json` | `feature_dim == 1021` |
| 输入帧 | 仍是整帧 `34*21 + LineData`；**截边在特征里做**，不是采集端裁掉 |

同事若自己拼了 1127 维全网格特征去喂新模型，也会失败；必须走当前 `TpFeatureExtractor`。

---

## 一句话回复同事

> 你说得对：rawfile 里那份是旧的 1127 维全网格模型。App 代码已经改成长边 3 层（1021 维），模型文件没同步，所以会 dim mismatch 全挂。请确认 `tp_model.json` 的 `feature_dim` 是 **1021** 且带 `tp_edge_only`，不要再用 1127 那份。
