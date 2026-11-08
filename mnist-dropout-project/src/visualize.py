"""Turn measured dropout runs into figures. Lines use paired runs on the same split."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gallery(images, labels, indices, folder, name, title, predictions=None):
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.8))
    fig.suptitle(title)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, indices):
        ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
        caption = str(int(labels[i]))
        if predictions is not None:
            caption += f" → {predictions[i]}"
        ax.set_title(caption)
    save(fig, folder, name)


def overview(x, y, train, val, folder):
    indices = [train[np.flatnonzero(y[train].numpy() == d)[0]] for d in range(10)]
    gallery(x, y, indices, folder, "examples", "One training example per digit")
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, label, offset in [(train, "Training", -.2), (val, "Validation", .2)]:
        ax.bar(np.arange(10)+offset, np.bincount(y[idx]), width=.4, label=label)
    ax.set(xticks=range(10), xlabel="Digit", ylabel="Images", title="Stratified development split")
    ax.legend()
    save(fig, folder, "distribution")


def comparisons(history, runs, folder):
    primary = runs[runs["size"] == 50000]
    h = history[history["size"] == 50000]
    for metric, label in [("loss", "Cross-entropy"), ("acc", "Accuracy (%)")]:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
        for ax, (rate, rows) in zip(axes, h.groupby("rate")):
            means = rows.groupby("epoch").mean(numeric_only=True)
            for part in ["train", "val"]:
                ax.plot(means.index, means[f"{part}_{metric}"] * (100 if metric == "acc" else 1), label=part)
            ax.set(title=f"Drop probability {rate:g}", xlabel="Epoch")
        axes[0].set_ylabel(label)
        axes[-1].legend()
        fig.suptitle(f"{label} by epoch · mean across seeds")
        save(fig, folder, f"learning_{metric}")
    means = primary.groupby("rate")
    metrics = ["train_acc", "val_acc", "test_acc"]
    ax = (100*means[metrics].mean()).plot.bar(yerr=100*means[metrics].std(), capsize=3, figsize=(8, 4))
    ax.set(xlabel="Drop probability", ylabel="Accuracy (%)", title="Selected checkpoints · mean ± seed SD",
           ylim=(max(0, 100*primary[metrics].min().min()-1), 100.6))
    ax.legend(["Training", "Validation", "Test"])
    save(ax.figure, folder, "comparison")
    for metric, name, ylabel in [("gap", "gap", "Train − validation accuracy (pp)"),
                                  ("val_acc", "sensitivity", "Validation accuracy (%)")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(means.size().index, means[metric].mean()*100, yerr=means[metric].std()*100, marker="o", capsize=4)
        ax.set(xlabel="Hidden-unit drop probability", ylabel=ylabel, title="Selected checkpoints · mean ± seed SD",
               xticks=means.size().index)
        save(fig, folder, name)
    fig, ax = plt.subplots(figsize=(7, 4))
    paired = runs[runs.seed == runs.seed.iloc[0]]
    for rate, rows in paired.groupby("rate"):
        rows = rows.sort_values("size")
        ax.plot(rows["size"], rows.val_acc*100, marker="o", label=f"Drop {rate:g}")
    ax.set(xlabel="Training images", ylabel="Validation accuracy (%)", title=f"Nested training cohorts · seed {paired.seed.iloc[0]}",
           xticks=sorted(paired["size"].unique()))
    ax.legend()
    save(fig, folder, "size")


def diagnostic(cm, pred, confidence, x, y, folder, rate):
    caption = f"Drop {rate:g} · seed 42"
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(xlabel="Predicted digit", ylabel="True digit", title=f"Test confusion counts · {caption}")
    save(fig, folder, "confusion")
    correct = pred == y.numpy()
    rng = np.random.default_rng(42)
    for mask, name in [(correct, "correct"), (~correct, "misclassified")]:
        idx = rng.permutation(np.flatnonzero(mask))[:10]
        gallery(x, y, idx, folder, name, f"{name.title()} · true → predicted · {caption}", pred)
    fig, ax = plt.subplots(figsize=(7, 4))
    for mask, label in [(correct, "Correct"), (~correct, "Incorrect")]:
        ax.hist(confidence[mask], bins=np.linspace(0, 1, 26), histtype="step", linewidth=2, label=label)
    ax.set(xlabel="Highest softmax probability", ylabel="Examples (log scale)", yscale="log", title=f"Test prediction confidence · {caption}")
    ax.legend()
    save(fig, folder, "confidence")
