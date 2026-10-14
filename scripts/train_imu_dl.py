"""Train IMU 1D-CNN at 100/200/300 ms windows. L2R/R2L onset-cut; OTHERS uncut."""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from grip_dl.imu_clip import (
    LABELS,
    RATE_HZ,
    build_xy,
    build_xy_recs,
    collect_imu_recs,
    label_from_name,
    list_imu_files,
    split_files,
    split_recs,
)
from grip_dl.imu_model import ImuWinNet, count_params


class ImuWinDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.ascontiguousarray(X, dtype=np.float32)  # (N, T, 6)
        self.y = np.ascontiguousarray(y, dtype=np.int64)

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, i):
        x = self.X[i]
        s = float(np.std(x))
        if s > 1e-6:
            x = x / s
        # (C, T) for Conv1d
        t = torch.from_numpy(np.ascontiguousarray(x.T))
        return t, torch.tensor(self.y[i], dtype=torch.long)


def make_loader(X, y, batch_size, shuffle, weighted=False, num_workers=0):
    ds = ImuWinDataset(X, y)
    sampler = None
    if weighted and len(ds) > 0:
        counts = np.bincount(y, minlength=3).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        w = (counts.sum() / (3.0 * counts))[y]
        sampler = WeightedRandomSampler(torch.from_numpy(w.astype(np.float32)), len(ds), replacement=True)
        shuffle = False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        ys.append(y.numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.zeros((0,), np.int64)
    y_pred = np.concatenate(ps) if ps else np.zeros((0,), np.int64)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    return acc, y_true, y_pred


def train_one(window_ms: int, data_root: Path, out_dir: Path, device, epochs=40, patience=8):
    files = list_imu_files(data_root)
    train_files, test_files = split_files(files, seed=42)
    print(f"\n=== window={window_ms}ms  rate={RATE_HZ}Hz stride=10ms ===", flush=True)
    print("train files:", [p.name for p in train_files if label_from_name(p) != "others"])
    print("test files:", [p.name for p in test_files if label_from_name(p) != "others"])
    print("others: train time 0-70%, test 70-100% of the same clip")
    Xtr, ytr, _ = build_xy(train_files, window_ms, 10.0, others_frac=(0.0, 0.70))
    Xte, yte, rec_te = build_xy(test_files, window_ms, 10.0, others_frac=(0.70, 1.0))
    return fit_eval(window_ms, Xtr, ytr, Xte, yte, rec_te, out_dir, device, epochs, patience)


def train_mix(window_ms: int, project_root: Path, out_dir: Path, device, epochs=40, patience=8):
    recs = collect_imu_recs(project_root)
    train_recs, test_recs = split_recs(recs, seed=42)
    print(f"\n=== mix window={window_ms}ms  rate={RATE_HZ}Hz stride=10ms ===", flush=True)
    print("old train:", [r.name for r in train_recs if r.protocol == "old" and label_from_name(r.path) != "others"])
    print("old test:", [r.name for r in test_recs if r.protocol == "old" and label_from_name(r.path) != "others"])
    print("twophase train:", [r.name for r in train_recs if r.protocol == "twophase" and label_from_name(r.path) != "others"])
    print("twophase test:", [r.name for r in test_recs if r.protocol == "twophase" and label_from_name(r.path) != "others"])
    print("others files: train 0-70%, test 70-100%; twophase swipe files: front=swipe (onset-cut), back=others")
    Xtr, ytr, _ = build_xy_recs(train_recs, window_ms, 10.0, others_frac=(0.0, 0.70))
    Xte, yte, rec_te = build_xy_recs(test_recs, window_ms, 10.0, others_frac=(0.70, 1.0))
    metrics = fit_eval(window_ms, Xtr, ytr, Xte, yte, rec_te, out_dir, device, epochs, patience)
    metrics["n_old"] = sum(1 for r in recs if r.protocol == "old")
    metrics["n_twophase"] = sum(1 for r in recs if r.protocol == "twophase")
    (out_dir / f"imu_win_{window_ms}ms_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def fit_eval(window_ms, Xtr, ytr, Xte, yte, rec_te, out_dir, device, epochs, patience):
    print(
        f"train={len(ytr)} {np.bincount(ytr, minlength=3).tolist()} "
        f"test={len(yte)} {np.bincount(yte, minlength=3).tolist()} shape={Xtr.shape}",
        flush=True,
    )
    if len(ytr) == 0 or len(yte) == 0:
        raise RuntimeError("empty split")

    train_loader = make_loader(Xtr, ytr, 64, True, weighted=True)
    test_loader = make_loader(Xte, yte, 128, False)
    counts = np.bincount(ytr, minlength=3).astype(np.float32)
    w = counts.sum() / (3.0 * np.maximum(counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    model = ImuWinNet(n_ch=6, n_classes=3).to(device)
    print(f"params={count_params(model):,} class_weight={w.tolist()}", flush=True)
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    ckpt = out_dir / f"imu_win_{window_ms}ms.pt"
    best_acc, bad, history = -1.0, 0, []

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * int(y.size(0))
            n += int(y.size(0))
        sched.step()
        acc, y_true, y_pred = eval_loop(model, test_loader, device)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "test_acc": acc})
        print(
            f"  ep {epoch:02d}/{epochs}  loss={total/max(n,1):.4f}  acc={acc:.4f}",
            flush=True,
        )
        if acc >= best_acc + 1e-4:
            best_acc, bad = acc, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "acc": acc,
                    "epoch": epoch,
                    "window_ms": window_ms,
                    "labels": LABELS,
                },
                ckpt,
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at {epoch}", flush=True)
                break

    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    acc, y_true, y_pred = eval_loop(model, test_loader, device)
    report = classification_report(
        y_true, y_pred, target_names=LABELS, digits=4, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4, zero_division=0))
    print("cm", cm, flush=True)

    # per-file majority
    by = defaultdict(lambda: {"y": [], "p": []})
    for rec, yt, yp in zip(rec_te, y_true, y_pred):
        by[rec]["y"].append(int(yt))
        by[rec]["p"].append(int(yp))
    by_file = {}
    file_ok = 0
    for rec, d in sorted(by.items()):
        true_lab = LABELS[int(np.bincount(d["y"]).argmax())]
        pred_lab = LABELS[int(np.bincount(d["p"]).argmax())]
        n = len(d["y"])
        wacc = float(np.mean(np.array(d["y"]) == np.array(d["p"])))
        by_file[rec] = {"true": true_lab, "pred_vote": pred_lab, "n": n, "win_acc": wacc}
        ok = pred_lab == true_lab
        file_ok += int(ok)
        print(f"  {rec:42} true={true_lab:6} vote={pred_lab:6} n={n:4} win_acc={wacc:.1%} {'OK' if ok else 'BAD'}")
    file_acc = file_ok / max(len(by_file), 1)

    metrics = {
        "window_ms": window_ms,
        "window_acc": acc,
        "file_vote_acc": file_acc,
        "best_epoch": int(blob.get("epoch", -1)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "counts_train": np.bincount(ytr, minlength=3).tolist(),
        "counts_test": np.bincount(yte, minlength=3).tolist(),
        "classification_report": report,
        "confusion_matrix": cm,
        "by_file": by_file,
        "history": history,
        "n_params": count_params(model),
    }
    (out_dir / f"imu_win_{window_ms}ms_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}", flush=True)
    root = ROOT
    out_dir = root / "artifacts" / "exp_imu_dl_win_mix100_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    m = train_mix(100, root, out_dir, device)
    print("\n=== mix 100ms ===")
    print(
        f"  win_acc={m['window_acc']:.4f}  file_vote={m['file_vote_acc']:.4f}  "
        f"best_ep={m['best_epoch']}  n_old={m.get('n_old')}  n_twophase={m.get('n_twophase')}"
    )


if __name__ == "__main__":
    main()
