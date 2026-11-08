"""Evaluate frozen validation-selected models and export measured evidence for class."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .dataset import cohorts, load_split
from .utils import config, load_model, run_name
from . import visualize as viz


@torch.inference_mode()
def predict(model, images):
    model.eval()
    probs = torch.cat([model(batch).softmax(1) for batch in images.split(512)])
    confidence, pred = probs.max(1)
    return pred.numpy(), confidence.numpy()


def metrics(labels, pred):
    digits = np.arange(10)
    return (confusion_matrix(labels, pred, labels=digits),
            classification_report(labels, pred, labels=digits, output_dict=True, zero_division=0))


def main():
    torch.set_num_threads(4)
    cfg = config()
    figures, tables = Path("results/figures"), Path("results/tables")
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    runs, histories = [], []
    # Require all planned histories before opening the holdout.
    for size in sorted(cfg["sizes"], reverse=True):
        for seed in cfg["seeds"] if size == 50000 else cfg["seeds"][:1]:
            for rate in cfg["rates"]:
                name = run_name(size, seed, rate)
                folder = Path("checkpoints") / name
                meta = json.loads((folder / "run.json").read_text())
                if meta["config"] != cfg:
                    raise ValueError(f"Stale settings in {folder}")
                h = pd.read_csv(folder / "history.csv").assign(size=size, seed=seed, rate=rate)
                selected = h.loc[h.epoch == meta["best_epoch"]].iloc[0].to_dict()
                runs.append(dict(selected, name=name, seconds=meta["seconds"], device=meta["device"]))
                histories.append(h)
    runs = pd.DataFrame(runs)
    history = pd.concat(histories, ignore_index=True)
    primary = runs[runs["size"] == 50000]
    focus = primary.groupby("rate").val_loss.mean().idxmin()
    test_x, test_y = load_split("t10k")
    errors = []
    for i, row in primary.iterrows():
        model = load_model(Path("checkpoints") / row["name"] / "best.pt")
        pred, confidence = predict(model, test_x)
        cm, report = metrics(test_y, pred)
        runs.loc[i, "test_acc"] = report["accuracy"]
        pd.DataFrame(cm, index=range(10), columns=range(10)).rename_axis("true_digit").to_csv(tables / f"{row['name']}_confusion.csv")
        pd.DataFrame({str(d): report[str(d)] for d in range(10)}).T.rename_axis("digit").to_csv(tables / f"{row['name']}_classification.csv")
        if row.rate == focus and row.seed == cfg["seeds"][0]:
            viz.diagnostic(cm, pred, confidence, test_x, test_y, figures, focus)
            np.fill_diagonal(cm, 0)
            for index in np.argsort(cm.ravel())[-3:][::-1]:
                a, b = np.unravel_index(index, cm.shape)
                errors.append(dict(actual=int(a), predicted=int(b), count=int(cm[a,b])))
    x, y = load_split()
    subsets, val = cohorts(y, cfg["sizes"], cfg["split_seed"])
    viz.overview(x, y, subsets[50000], val, figures)
    viz.comparisons(history, runs, figures)
    history.to_csv(tables / "epochs.csv", index=False)
    runs.to_csv(tables / "runs.csv", index=False)
    cols = ["train_loss","val_loss","train_acc","val_acc","test_acc","gap","seconds"]
    summary = runs[runs["size"] == 50000].groupby("rate")[cols].agg(["mean","std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(tables / "primary.csv")
    data = dict(config=cfg, primary=summary.reset_index().to_dict("records"),
                runs=json.loads(runs.to_json(orient="records")), diagnostic_rate=focus,
                diagnostic_seed=cfg["seeds"][0], common_errors=errors,
                figures=[str(p) for p in sorted(figures.glob("*.png"))])
    Path("presentation/presentation_data.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    make_summary(summary, runs, errors, focus, history)


def make_summary(summary, runs, errors, focus, history):
    lines = ["# Experimental results", "", "All primary entries are means ± sample SD over seeds 42, 43 and 44. Accuracy is a percentage. The gap is measured in percentage points. Metrics use each run’s lowest-validation-loss epoch.", "",
             "| Drop | Train accuracy | Validation accuracy | Test accuracy | Gap | Train loss | Validation loss | Seconds |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rate, row in summary.iterrows():
        cells = []
        for metric in ["train_acc","val_acc","test_acc","gap","train_loss","val_loss","seconds"]:
            scale = 100 if metric.endswith("acc") or metric=="gap" else 1
            cells.append(f"{row[metric+'_mean']*scale:.3f} ± {row[metric+'_std']*scale:.3f}")
        lines.append("| " + " | ".join([str(rate), *cells]) + " |")
    lines += ["", "## Dataset size", "", "Seed 42 validation accuracy (%) at the selected checkpoint.", "",
              "| Images | Drop 0 | Drop 0.2 | Drop 0.5 |", "|---:|---:|---:|---:|"]
    for size, rows in runs[runs.seed == 42].groupby("size"):
        lines.append(f"| {int(size)} | " + " | ".join(f"{100*v:.2f}" for v in rows.sort_values("rate").val_acc) + " |")
    lines += ["", f"The diagnostic model uses drop probability {focus:g}, selected by mean validation loss, and predefined seed 42. Its most frequent directional errors on the 10,000 test digits were " + ", ".join(f"{e['actual']} → {e['predicted']} ({e['count']})" for e in errors) + ".", ""]
    best = summary.loc[focus]
    baseline = summary.loc[0.0]
    lines += ["## Interpretation", "",
              f"The lowest mean validation loss occurred at drop probability {focus:g}. "
              f"Compared with the baseline, this rate changed mean training accuracy by "
              f"{100*(best.train_acc_mean-baseline.train_acc_mean):+.2f} pp, validation accuracy by "
              f"{100*(best.val_acc_mean-baseline.val_acc_mean):+.2f} pp, and test accuracy by "
              f"{100*(best.test_acc_mean-baseline.test_acc_mean):+.2f} pp. "
              f"The gap changed from {100*baseline.gap_mean:.2f} to {100*best.gap_mean:.2f} pp.", ""]
    final = history[history["size"] == 50000].groupby(["seed","rate"]).tail(1)
    terminal = final.groupby("rate")[["train_acc","val_acc","gap","val_loss"]].mean()
    lines += ["### At the end of the fixed training budget", "",
              "| Drop | Train accuracy (%) | Validation accuracy (%) | Gap (pp) | Validation loss |",
              "|---:|---:|---:|---:|---:|"]
    for rate, row in terminal.iterrows():
        lines.append(f"| {rate:g} | {row.train_acc*100:.2f} | {row.val_acc*100:.2f} | "
                     f"{row.gap*100:.2f} | {row.val_loss:.4f} |")
    lines += ["", "Compare these endpoint gaps with the selected-checkpoint gaps above. "
              "The validation-loss minimum can occur at a different epoch for each rate.", ""]
    paired = runs[runs.seed == 42].pivot(index="size", columns="rate", values="val_acc")
    gains = (paired[0.5] - paired[0.0])*100
    lines += ["At seed 42, switching from zero to 0.5 dropout changed validation accuracy by "
              + ", ".join(f"{value:+.2f} pp at {int(n):,} images" for n, value in gains.items())
              + ". These comparisons use equal epochs, so the larger cohorts also receive more updates.", "",
              "The three primary seeds share one validation split and one test set. "
              "The seed SD describes optimizer and initialization variability. The size study has one seed.", ""]
    Path("presentation/experiment_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
