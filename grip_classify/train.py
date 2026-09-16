"""Train RandomForest classifiers for IMU and TP grip-state prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import ID_TO_LABEL, LABELS, WINDOW_MS
from .data import load_all_recordings
from .features import extract_imu_windows, extract_tp_windows, make_imu_dataset, make_tp_dataset


def _build_clf(n_estimators: int = 300, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=24,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _eval(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "accuracy": float(report["accuracy"]),
    }


def _class_counts(y: np.ndarray) -> Dict[str, int]:
    counts = {name: 0 for name in LABELS}
    for i, name in enumerate(LABELS):
        counts[name] = int(np.sum(y == i))
    return counts


def _xy_for_split(
    modality: str,
    recordings: Dict,
    holdout: Optional[Dict],
    window_ms: float,
    robust_p2_p98: bool = True,
    window_cumsum: bool = True,
    edge_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if modality == "imu":
        if holdout:
            Xtr, ytr = extract_imu_windows(recordings, window_ms=window_ms)
            Xte, yte = extract_imu_windows(holdout, window_ms=window_ms)
            return Xtr, ytr, Xte, yte, "holdout"
        return (*make_imu_dataset(recordings, window_ms=window_ms), "temporal_80_20")
    if holdout:
        Xtr, ytr = extract_tp_windows(
            recordings,
            window_ms=window_ms,
            robust_p2_p98=robust_p2_p98,
            window_cumsum=window_cumsum,
            edge_only=edge_only,
        )
        Xte, yte = extract_tp_windows(
            holdout,
            window_ms=window_ms,
            robust_p2_p98=robust_p2_p98,
            window_cumsum=window_cumsum,
            edge_only=edge_only,
        )
        return Xtr, ytr, Xte, yte, "holdout"
    Xtr, ytr, Xte, yte = make_tp_dataset(
        recordings,
        window_ms=window_ms,
        robust_p2_p98=robust_p2_p98,
        window_cumsum=window_cumsum,
    )
    return Xtr, ytr, Xte, yte, "temporal_80_20"


def train_imu_model(
    data_root: Path,
    out_dir: Path,
    window_ms: float = WINDOW_MS,
    recordings: Optional[Dict] = None,
    holdout_recordings: Optional[Dict] = None,
) -> Tuple[Pipeline, Dict]:
    if recordings is None:
        recordings = load_all_recordings(data_root)
    Xtr, ytr, Xte, yte, split = _xy_for_split("imu", recordings, holdout_recordings, window_ms)
    print(
        f"[imu] split={split} train={len(ytr)} { _class_counts(ytr) } "
        f"test={len(yte)} { _class_counts(yte) } feat_dim={Xtr.shape[1]}"
    )
    clf = _build_clf()
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    metrics = _eval(yte, pred)
    metrics["split"] = split
    print(classification_report(yte, pred, target_names=LABELS, digits=4, zero_division=0))
    print("confusion_matrix:\n", confusion_matrix(yte, pred, labels=list(range(len(LABELS)))))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "imu_model.joblib"
    joblib.dump(
        {
            "pipeline": clf,
            "labels": LABELS,
            "id_to_label": ID_TO_LABEL,
            "window_ms": window_ms,
            "modality": "imu",
            "feature_dim": int(Xtr.shape[1]),
            "eval_split": split,
        },
        model_path,
    )
    with open(out_dir / "imu_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[imu] saved {model_path} acc={metrics['accuracy']:.4f}")
    return clf, metrics


def train_tp_model(
    data_root: Path,
    out_dir: Path,
    window_ms: float = WINDOW_MS,
    recordings: Optional[Dict] = None,
    holdout_recordings: Optional[Dict] = None,
    robust_p2_p98: bool = True,
    window_cumsum: bool = True,
    name_suffix: str = "",
    edge_only: bool = True,
) -> Tuple[Pipeline, Dict]:
    if recordings is None:
        recordings = load_all_recordings(data_root)
    tag = "p2-p98" if robust_p2_p98 else "no-norm"
    print(
        f"[tp] robust_p2_p98={robust_p2_p98} ({tag}) "
        f"window_cumsum={window_cumsum} edge_only={edge_only}"
    )
    Xtr, ytr, Xte, yte, split = _xy_for_split(
        "tp",
        recordings,
        holdout_recordings,
        window_ms,
        robust_p2_p98=robust_p2_p98,
        window_cumsum=window_cumsum,
        edge_only=edge_only,
    )
    print(
        f"[tp] split={split} train={len(ytr)} { _class_counts(ytr) } "
        f"test={len(yte)} { _class_counts(yte) } feat_dim={Xtr.shape[1]}"
    )
    clf = _build_clf(n_estimators=400)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    metrics = _eval(yte, pred)
    metrics["split"] = split
    metrics["robust_p2_p98"] = robust_p2_p98
    metrics["window_cumsum"] = window_cumsum
    metrics["tp_edge_only"] = bool(edge_only)
    print(classification_report(yte, pred, target_names=LABELS, digits=4, zero_division=0))
    print("confusion_matrix:\n", confusion_matrix(yte, pred, labels=list(range(len(LABELS)))))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"tp_model{name_suffix}.joblib"
    joblib.dump(
        {
            "pipeline": clf,
            "labels": LABELS,
            "id_to_label": ID_TO_LABEL,
            "window_ms": window_ms,
            "modality": "tp",
            "feature_dim": int(Xtr.shape[1]),
            "uses_cumsum_absolute": bool(window_cumsum),
            "window_cumsum_only": bool(window_cumsum),
            "window_demean": True,
            "tp_edge_only": bool(edge_only),
            "robust_p2_p98": robust_p2_p98,
            "eval_split": split,
        },
        model_path,
    )
    metrics_path = out_dir / f"tp_metrics{name_suffix}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[tp] saved {model_path} acc={metrics['accuracy']:.4f}")
    return clf, metrics
