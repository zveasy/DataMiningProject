"""Fit paired dropout treatments. Select the best epoch using validation loss alone."""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from .dataset import cohorts, load_split
from .model import DigitNet
from .utils import config, device, run_name, seed_all


@torch.inference_mode()
def score(model, x, y, batch=1024):
    model.eval()
    loss = x.new_zeros(())
    correct = x.new_zeros(())
    for xb, yb in zip(x.split(batch), y.split(batch)):
        logits = model(xb)
        loss += F.cross_entropy(logits, yb, reduction="sum")
        correct += (logits.argmax(1) == yb).sum()
    return loss.item() / len(y), correct.item() / len(y)


def fit(train, valid, cfg, seed, rate, folder, target):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = DigitNet(rate, cfg["hidden"]).to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    order_rng = torch.Generator().manual_seed(seed)
    x, y = (t.to(target) for t in train)
    valid = tuple(t.to(target) for t in valid)
    best, best_epoch = float("inf"), 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = torch.randperm(len(y), generator=order_rng).to(target)
        for idx in order.split(cfg["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
        train_loss, train_acc = score(model, x, y)
        val_loss, val_acc = score(model, *valid)
        row = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                   val_loss=val_loss, val_acc=val_acc, gap=train_acc-val_acc)
        history.append(row)
        checkpoint = dict(weights=model.state_dict(), hidden=cfg["hidden"], rate=rate, epoch=epoch)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            torch.save(checkpoint, folder / "best.pt")
        if cfg["patience"] and epoch-best_epoch >= cfg["patience"]:
            break
    torch.save(checkpoint, folder / "last.pt")
    pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    meta = dict(config=cfg, seed=seed, rate=rate, size=len(y), best_epoch=best_epoch,
                seconds=time.perf_counter()-start, device=target, torch=str(torch.__version__))
    (folder / "run.json").write_text(json.dumps(meta, indent=2))
    print(folder.name, best_epoch, f"{meta['seconds']:.1f}s", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=device())
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = config()
    x, y = load_split()
    subsets, val = cohorts(y, cfg["sizes"], cfg["split_seed"])
    for size in sorted(subsets, reverse=True):
        indices = subsets[size]
        train = x[indices], y[indices]
        for seed in cfg["seeds"] if size == 50000 else cfg["seeds"][:1]:
            for rate in cfg["rates"]:
                folder = Path("checkpoints") / run_name(size, seed, rate)
                fit(train, (x[val], y[val]), cfg, seed, rate, folder, args.device)


if __name__ == "__main__":
    main()
