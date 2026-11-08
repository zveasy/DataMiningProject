"""Decode Kaggle IDX files and build shared stratified cohorts for dropout comparisons."""
import gzip
import struct
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def read_idx(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        raw = stream.read()
    if len(raw) < 8:
        raise ValueError("Truncated IDX header")
    magic, count = struct.unpack(">II", raw[:8])
    if magic == 2051 and raw[8:16] == struct.pack(">II", 28, 28):
        shape, offset = (count, 1, 28, 28), 16
    elif magic == 2049:
        shape, offset = (count,), 8
    else:
        raise ValueError("Unexpected IDX magic or image dimensions")
    if len(raw) - offset != int(np.prod(shape)) or not count:
        raise ValueError("Incorrect IDX payload length")
    return torch.from_numpy(np.frombuffer(raw[offset:], dtype=np.uint8).reshape(shape).copy())


def locate(folder, stem):
    for suffix in ("", ".gz"):
        for name in (stem, stem.replace(".idx", "-idx")):
            for match in sorted(Path(folder).rglob(name + suffix)):
                if match.is_file():
                    return match
    raise FileNotFoundError(f"Missing {stem} in {folder}. Extract the hojjatk/mnist-dataset Kaggle archive into data/.")


def load_split(part="train", folder="data"):
    expected = {"train": 60000, "t10k": 10000}[part]
    images = read_idx(locate(folder, part + "-images.idx3-ubyte"))
    labels = read_idx(locate(folder, part + "-labels.idx1-ubyte")).long()
    if images.shape != (expected, 1, 28, 28) or labels.shape != (expected,) or set(labels.unique().tolist()) != set(range(10)):
        raise ValueError("Invalid MNIST dimensions or labels")
    return normalize(images), labels


def normalize(images):
    return images.float().div_(255)


def stratified(indices, labels, size, seed):
    train, rest = train_test_split(indices, train_size=size, stratify=labels[indices], random_state=seed)
    return train, rest


def cohorts(labels, sizes, seed=42, validation_size=10000):
    train, val = stratified(np.arange(len(labels)), labels, len(labels)-validation_size, seed)
    subsets = {}
    for size in sorted(sizes, reverse=True):
        if size != len(train):
            train, _ = stratified(train, labels, size, seed)
        subsets[size] = torch.from_numpy(train)
    return subsets, torch.from_numpy(val)
