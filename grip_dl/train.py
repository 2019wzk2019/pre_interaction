"""Train / evaluate TpGridNet on full 34×21 TP diffs.

Default: dry-run (shape + ONNX only). Pass --train on a GPU box to fit.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from grip_classify import LABELS
from grip_classify.data import load_all_recordings, load_holdout_recordings
from grip_dl import EDGE_ONLY, GRID_H, GRID_W, N_CLASSES
from grip_dl.config import TrainConfig
from grip_dl.dataset import dummy_batch, make_loader
from grip_dl.export import export_onnx
from grip_dl.model import TpGridNet, count_params
from grip_dl.windows import extract_tp_sequences


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dry_run(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model = TpGridNet(n_frames=cfg.n_frames).to(device)
    model.eval()
    x, y = dummy_batch(batch_size=4, n_frames=cfg.n_frames)
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    print("=== TP DL dry-run (full 34x21, not training) ===")
    print(f"input        {tuple(x.shape)}  # (B, T={cfg.n_frames}, H={GRID_H}, W={GRID_W})")
    print(f"logits       {tuple(logits.shape)}")
    print(f"loss         {float(loss):.4f}")
    print(f"params       {count_params(model):,}")
    print(f"window_ms    {cfg.window_ms}")
    print(f"edge_only    {EDGE_ONLY}")
    print(f"device       {device}")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = cfg.out_dir / "tp_grid_net_dryrun.onnx"
    try:
        export_onnx(model.cpu(), onnx_path, n_frames=cfg.n_frames)
        print(f"onnx         {onnx_path}  ({onnx_path.stat().st_size} bytes)")
    except Exception as exc:
        print(f"onnx         skipped ({exc})")
    meta = {
        "status": "dry-run",
        "window_ms": cfg.window_ms,
        "n_frames": cfg.n_frames,
        "grid_h": GRID_H,
        "grid_w": GRID_W,
        "edge_only": EDGE_ONLY,
        "n_params": count_params(model),
        "labels": list(LABELS),
        "input_layout": f"(B, T, H, W) = (B, {cfg.n_frames}, {GRID_H}, {GRID_W})",
        "trained": False,
    }
    (cfg.out_dir / "dryrun.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _loop(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total = 0.0
    n = 0
    ys, preds = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = int(y.size(0))
            total += float(loss.item()) * bs
            n += bs
            ys.append(y.detach().cpu().numpy())
            preds.append(logits.argmax(dim=1).detach().cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(preds) if preds else np.zeros((0,), dtype=np.int64)
    return total / max(n, 1), y_true, y_pred


def _eval_folders(model, holdout, cfg, device) -> dict:
    model.eval()
    by_folder = {}
    for name, rec in sorted(holdout.items()):
        X, y = extract_tp_sequences(
            {name: rec},
            window_ms=cfg.window_ms,
            n_frames=cfg.n_frames,
            edge_only=EDGE_ONLY,
        )
        if len(y) == 0:
            print(f"  {name}: no windows")
            continue
        loader = make_loader(X, y, cfg.batch_size, shuffle=False)
        _, y_true, y_pred = _loop(model, loader, None, nn.CrossEntropyLoss(), device, train=False)
        counts = {LABELS[i]: int(np.sum(y_pred == i)) for i in range(len(LABELS))}
        acc = float(np.mean(y_pred == y_true))
        by_folder[name] = {
            "true_label": rec["label"],
            "n": int(len(y)),
            "acc": acc,
            "pred_counts": counts,
        }
        print(f"  {name:8} true={rec['label']:6} n={len(y):4} acc={acc:.1%}  pred={counts}", flush=True)
    return by_folder


def run_train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(
        f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()}",
        flush=True,
    )
    recordings = load_all_recordings(cfg.train_root)
    holdout = load_holdout_recordings(cfg.test_root)
    print("extracting train windows (full 34x21)...", flush=True)
    Xtr, ytr = extract_tp_sequences(
        recordings, window_ms=cfg.window_ms, n_frames=cfg.n_frames, edge_only=EDGE_ONLY
    )
    del recordings
    print("extracting test windows...", flush=True)
    Xte, yte = extract_tp_sequences(
        holdout, window_ms=cfg.window_ms, n_frames=cfg.n_frames, edge_only=EDGE_ONLY
    )
    print(
        f"train={len(ytr)} test={len(yte)} shape={Xtr.shape} "
        f"counts_tr={np.bincount(ytr, minlength=3).tolist()} "
        f"counts_te={np.bincount(yte, minlength=3).tolist()}",
        flush=True,
    )

    train_loader = make_loader(
        Xtr, ytr, cfg.batch_size, shuffle=True, weighted=True, num_workers=cfg.num_workers
    )
    test_loader = make_loader(Xte, yte, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    counts = np.bincount(ytr, minlength=N_CLASSES).astype(np.float32)
    w = counts.sum() / (N_CLASSES * np.maximum(counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    model = TpGridNet(n_frames=cfg.n_frames).to(device)
    print(f"class_weight={w.tolist()} params={count_params(model):,}", flush=True)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    history = []
    ckpt_path = cfg.out_dir / "tp_grid_net.pt"

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, _, _ = _loop(model, train_loader, opt, criterion, device, train=True)
        te_loss, y_true, y_pred = _loop(model, test_loader, opt, criterion, device, train=False)
        sched.step()
        acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "test_loss": te_loss,
                "test_acc": acc,
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        print(
            f"epoch {epoch:03d}/{cfg.epochs}  train_loss={tr_loss:.4f}  "
            f"test_loss={te_loss:.4f}  acc={acc:.4f}  lr={opt.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "acc": acc,
                    "epoch": epoch,
                    "n_frames": cfg.n_frames,
                    "grid_h": GRID_H,
                    "grid_w": GRID_W,
                    "window_ms": cfg.window_ms,
                    "edge_only": EDGE_ONLY,
                    "labels": list(LABELS),
                },
                ckpt_path,
            )
            try:
                export_onnx(model.cpu(), cfg.out_dir / "tp_grid_net.onnx", n_frames=cfg.n_frames)
            except Exception as exc:
                print(f"onnx export skipped ({exc})", flush=True)
            model.to(device)

    blob = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(blob["model"])
    te_loss, y_true, y_pred = _loop(model, test_loader, opt, criterion, device, train=False)
    report = classification_report(
        y_true, y_pred, target_names=LABELS, digits=4, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES))).tolist()
    print("confusion_matrix", cm, flush=True)
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4, zero_division=0))
    print("=== per test folder ===", flush=True)
    by_folder = _eval_folders(model, holdout, cfg, device)
    metrics = {
        "accuracy": float(report["accuracy"]),
        "best_epoch": int(blob.get("epoch", -1)),
        "classification_report": report,
        "confusion_matrix": cm,
        "history": history,
        "by_folder": by_folder,
        "window_ms": cfg.window_ms,
        "n_frames": cfg.n_frames,
        "grid_h": GRID_H,
        "grid_w": GRID_W,
        "edge_only": EDGE_ONLY,
        "device": str(device),
        "n_params": count_params(model),
    }
    (cfg.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved {cfg.out_dir} acc={metrics['accuracy']:.4f} best_epoch={metrics['best_epoch']}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-grid TP ConvNet (34x21).")
    p.add_argument("--train", action="store_true", help="load data and fit")
    p.add_argument("--train-root", type=Path, default=Path("data/tp/train"))
    p.add_argument("--test-root", type=Path, default=Path("data/tp/test"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts") / "exp_tp_dl_single")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-frames", type=int, default=1)
    p.add_argument("--window-ms", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        train_root=args.train_root,
        test_root=args.test_root,
        out_dir=args.out_dir,
        window_ms=args.window_ms,
        n_frames=args.n_frames,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        num_workers=args.num_workers,
        lr=args.lr,
    )
    if args.train:
        run_train(cfg)
    else:
        dry_run(cfg)


if __name__ == "__main__":
    main()
