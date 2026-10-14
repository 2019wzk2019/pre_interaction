最新纯 TP 模型的数据。类别：左to右 / 右to左 / others。
每段录音是一组 imu_continuous*.csv + diffdata_continuous*.jsonl + collection_run_metadata*.json。
模型只吃 jsonl 里的 34×21 原始一阶差分，单帧，不做 cumsum。

train/  121 段，用来训练
  左to右       55 段  -> 左to右
  右to左       61 段  -> 右to左
  others空置    4 段  -> others
  others点击    1 段  -> others

test/  2 段，用来留出评估，不参与训练
  c左to右  1 段 -> 左to右
  c右to左  1 段 -> 右to左

从仓库根目录重训（与 delivery/hap_delivery_DL 同一设定：1 帧、全屏、per-sample std）：

  pip install -r requirements-dl.txt
  python scripts/train_tp_dl.py --train --device auto --epochs 30 --batch-size 64

权重写到 artifacts/exp_tp_dl_single/。
旧的 500ms / 42 帧实验要显式加上 --n-frames 42 --window-ms 500。
data/wzk/ 是更早的随机森林数据，不要拿来重训这份单帧模型。
