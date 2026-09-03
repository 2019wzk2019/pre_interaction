# pre_interaction

Grip / reach pre-interaction classification from IMU and TP (touch panel).

Classes: `左to右` / `右to左` / `others`.

## Train

```bash
python train_grip_models.py --modality both --window-ms 150
```

## HarmonyOS app

Open `harmony_app/GripReachPredict` in DevEco Studio. Models live in `entry/src/main/resources/rawfile/`. Sync with:

```bash
python sync_harmony_models.py
```
