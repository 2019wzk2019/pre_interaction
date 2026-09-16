"""Deep-learning TP grip / reach classifier (full 34×21 grid)."""

from grip_classify import ID_TO_LABEL, LABEL_TO_ID, LABELS, TP_EDGE_LAYERS, TP_GRID_COLS, TP_GRID_ROWS

WINDOW_MS = 500.0
STRIDE_MS = 50.0
N_FRAMES = 42  # round(500 / 12ms); windows resampled to this length
GRID_H = TP_GRID_ROWS  # 34
GRID_W = TP_GRID_COLS  # 21
N_CELLS = GRID_H * GRID_W  # 714
N_CLASSES = len(LABELS)
EDGE_ONLY = False

__all__ = [
    "LABELS",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "WINDOW_MS",
    "STRIDE_MS",
    "N_FRAMES",
    "GRID_H",
    "GRID_W",
    "N_CELLS",
    "N_CLASSES",
    "EDGE_ONLY",
    "TP_GRID_ROWS",
    "TP_GRID_COLS",
    "TP_EDGE_LAYERS",
]
