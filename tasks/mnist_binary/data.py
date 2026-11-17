"""Read the canonical MNIST training split and make a binary holdout."""

import gzip
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
import torch


BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
TRAIN_FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
}


def training_file(directory: Path, name: str) -> bytes:
    path = directory / name
    if path.is_file():
        compressed = path.read_bytes()
    else:
        with urlopen(BASE_URL + name, timeout=30) as response:
            compressed = response.read()
    if hashlib.sha256(compressed).hexdigest() != TRAIN_FILES[name]:
        raise ValueError(f"Unexpected MNIST SHA-256 for {path}")
    if not path.exists():
        with TemporaryDirectory(dir=directory) as temporary:
            staged = Path(temporary) / name
            staged.write_bytes(compressed)
            staged.replace(path)
    return gzip.decompress(compressed)


def load_training_data(directory: Path, split_seed: int):
    directory.mkdir(parents=True, exist_ok=True)
    image_bytes, label_bytes = (training_file(directory, name) for name in TRAIN_FILES)
    images = np.frombuffer(image_bytes, dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8)
    rng = np.random.default_rng(split_seed)
    train = []
    validation = []
    for digit in (0, 1):
        indices = rng.permutation(np.flatnonzero(labels == digit))
        validation.extend(indices[:500])
        train.extend(indices[500:])
    train = np.sort(train)
    validation = np.sort(validation)
    indices = np.sort(np.concatenate([train, validation]))
    return (
        torch.from_numpy(images[indices]).unsqueeze(1).to(torch.float32) / 255,
        torch.from_numpy(labels[indices].astype(np.int64)),
        train,
        validation,
        torch.from_numpy(np.isin(indices, validation)),
    )
