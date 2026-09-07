"""Grip / reach state classification from IMU and TP (touch panel) streams."""

# Model classes: reach-left/right vs everything else (normal left/right grip).
LABELS = ["左to右", "右to左", "others"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

# On-disk folders under Wzk/. Static grips map to others.
SOURCE_FOLDERS = ["左to右", "左", "左1", "右to左", "右", "右1"]
FOLDER_TO_LABEL = {
    "左to右": "左to右",
    "右to左": "右to左",
    "左": "others",
    "左1": "others",
    "右": "others",
    "右1": "others",
}

WINDOW_MS = 150
STRIDE_MS = 50
HEAD_EXCLUDE_MS = 1000
PRE_TAP_EXCLUDE_MS = 1000
TAIL_EXCLUDE_MS = 3000
HEAD_EXCLUDE_FOLDERS = ("左to右", "右to左")
TRAIN_DIRNAME = "训练集"
HOLDOUT_DIRNAME = "测试集"
TP_GRID_ROWS = 34
TP_GRID_COLS = 21
# Portrait long edges = left/right columns; keep this many outer layers.
TP_EDGE_LAYERS = 3
