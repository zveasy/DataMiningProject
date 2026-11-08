import gzip
import struct

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import cohorts, load_split, normalize, read_idx
from src.evaluate import metrics, predict
from src.model import DigitNet
from src.train import fit, score
from src.utils import config, load_model, seed_all
from src.visualize import save


@pytest.mark.parametrize("compressed", [False, True])
def test_idx_images_and_labels(tmp_path, compressed):
    for magic, payload, shape, extra in [
        (2049, bytes([4, 9]), (2,), b""),
        (2051, bytes([255, 0])*784, (2, 1, 28, 28), struct.pack(">II", 28, 28)),
    ]:
        data = struct.pack(">II", magic, 2) + extra + payload
        path = tmp_path / ("digits.gz" if compressed else "digits")
        path.write_bytes(gzip.compress(data) if compressed else data)
        tensor = read_idx(path)
        assert tensor.shape == shape
        assert tensor.numpy().tobytes() == payload


@pytest.mark.parametrize("raw", [b"X", struct.pack(">II", 2049, 2)+b"0", struct.pack(">II", 2048, 1)])
def test_corrupt_idx(tmp_path, raw):
    path = tmp_path / "bad"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        read_idx(path)


def test_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_split(folder=tmp_path)


def test_normalization():
    t = normalize(torch.tensor([0, 127, 255], dtype=torch.uint8))
    torch.testing.assert_close(t, torch.tensor([0, 127/255, 1]))


def test_shared_stratified_subsets():
    labels = torch.arange(1000) % 10
    subsets, val = cohorts(labels, [100, 400, 900], validation_size=100)
    assert set(subsets[100].tolist()) < set(subsets[400].tolist()) < set(subsets[900].tolist())
    assert set(val.tolist()).isdisjoint(subsets[900].tolist())
    assert len(set(val.tolist()) | set(subsets[900].tolist())) == 1000
    assert torch.equal(val, cohorts(labels, [900], validation_size=100)[1])
    for n, idx in subsets.items():
        assert torch.all(labels[idx].bincount() == n//10)


def test_fit_and_reload(tmp_path):
    import pandas as pd
    torch.set_num_threads(2)
    cfg = dict(config(), epochs=4, hidden=[12], batch_size=8)
    x = torch.zeros(32,1,28,28)
    y = torch.ones(32, dtype=torch.long)
    seed_all(42)
    initial = DigitNet(.2, cfg["hidden"])
    final = fit((x,y), (x,y), cfg, 42, .2, tmp_path, "cpu")
    assert not torch.equal(initial[-1].bias, final[-1].bias)
    assert score(final,x,y)[0] < score(initial,x,y)[0]
    h = pd.read_csv(tmp_path / "history.csv")
    assert len(h) == 4
    assert score(load_model(tmp_path / "best.pt"),x,y)[0] == pytest.approx(h.val_loss.min())
    torch.testing.assert_close(load_model(tmp_path / "last.pt")(x), final(x))
    pred, conf = predict(final, x)
    assert pred.shape == conf.shape == (32,)
    assert np.all((conf >= 0.1) & (conf <= 1))


def test_evaluation_metrics():
    cm, report = metrics([1,1,2,2], [1,2,2,2])
    assert cm.shape == (10,10)
    assert cm[1,2] == 1
    assert report["accuracy"] == .75
    assert report["1"]["recall"] == .5
    assert report["2"]["precision"] == pytest.approx(2/3)


def test_png_resolution(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0,1], [1,0])
    save(fig, tmp_path, "curve")
    with Image.open(tmp_path / "curve.png") as image:
        assert image.format == "PNG"
        assert image.info["dpi"][0] == pytest.approx(300, abs=.1)


def test_early_stop_on_flat_validation_loss(tmp_path):
    import pandas as pd
    cfg = dict(config(), epochs=8, patience=2, learning_rate=0.0, hidden=[8])
    x = torch.zeros(16, 1, 28, 28)
    y = torch.zeros(16, dtype=torch.long)
    fit((x,y), (x,y), cfg, 42, .2, tmp_path, "cpu")
    h = pd.read_csv(tmp_path / "history.csv")
    assert h.epoch.tolist() == [1, 2, 3]
    assert torch.load(tmp_path / "best.pt", weights_only=True)["epoch"] == 1


def test_paired_initial_weights():
    seed_all(43)
    baseline = DigitNet(0)
    seed_all(43)
    regularized = DigitNet(.5)
    for left, right in zip(baseline.parameters(), regularized.parameters()):
        torch.testing.assert_close(left, right)
