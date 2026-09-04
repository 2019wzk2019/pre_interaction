# pre_interaction

Grip / reach pre-interaction classification from IMU and TP (touch panel).

Classes: `左to右` / `右to左` / `others`.

## Train

```bash
python train_grip_models.py --modality both --window-ms 150
```

## HarmonyOS app

HAP 工程 **不在本仓库**（避免把同事/厂商侧代码推到 GitHub）。适配说明见 `docs/HAP_TP_ADAPT.md`。

本地若仍保留 `harmony_app/`，可用下面命令把 joblib 导出成 json（只写本机，不会进 git）：

```bash
python sync_harmony_models.py
```
