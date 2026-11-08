"""Shared settings and checkpoint loading keep inference aligned with training."""
import json
from pathlib import Path
import random

import numpy as np
import torch

from .model import DigitNet


def config():
    return json.loads(Path("configs/experiment.json").read_text())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def run_name(size, seed, rate):
    return f"n{size}_seed{seed}_drop{rate:.1f}"


def load_model(path, target="cpu"):
    saved = torch.load(path, map_location=target, weights_only=True)
    model = DigitNet(saved["rate"], saved["hidden"]).to(target)
    model.load_state_dict(saved["weights"])
    return model.eval()
