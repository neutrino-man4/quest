"""
Dataloader for JetClass HDF5 datasets.
Supports train/val/test splits for binary signal-vs-background jet classification.
Returns PennyLane numpy tensors via a PyTorch DataLoader with a custom collate function.
Author: Aritra Bal (ETP)
Date: 2026-06-03
"""

import glob
import os
import warnings
from typing import Tuple, Union

import h5py
import numpy as np
import pennylane.numpy as pnp
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_split_dir(split: str, flatten: bool) -> str:
    """Map a logical split name to the actual directory prefix."""
    return f"flat_{split}" if flatten else split


def _get_sorted_files(base_path: str, split: str, jet_type: str, flatten: bool) -> list:
    pattern = os.path.join(base_path, _resolve_split_dir(split, flatten), jet_type, "*.h5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No h5 files found at: {pattern}")
    return files


def _load_jets(file_paths: list, n_jets: int, num_particles: int) -> np.ndarray:
    """
    Load up to n_jets from a sorted list of h5 files without per-jet loops.
    Reads each file in one vectorised h5py call and stops early once quota is met.
    Returns shape (n_loaded, num_particles, 3).
    """
    arrays = []
    total = 0
    for fpath in file_paths:
        if total >= n_jets:
            break
        with h5py.File(fpath, "r") as f:
            # single read: full file slice, vectorised
            chunk = f["jetConstituentsList"][:, :num_particles, :]  # (M, num_particles, 3)
        remaining = n_jets - total
        arrays.append(chunk[:remaining])
        total += min(len(chunk), remaining)

    if not arrays:
        raise RuntimeError("No data was loaded — check file paths and n_jets value.")

    loaded = np.concatenate(arrays, axis=0)  # (total, num_particles, 3)
    if total < n_jets:
        warnings.warn(
            f"Requested {n_jets} jets but only {total} were available across {len(file_paths)} file(s)."
        )
    return loaded.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JetDataset(Dataset):
    """PyTorch Dataset holding signal + background jet constituents and binary labels."""

    def __init__(self, sig_data: np.ndarray, bkg_data: np.ndarray) -> None:
        # label: 1 = signal, 0 = background
        self.data = np.concatenate([sig_data, bkg_data], axis=0)   # (N, P, 3)
        self.labels = np.concatenate([
            np.ones(len(sig_data), dtype=np.float32),
            np.zeros(len(bkg_data), dtype=np.float32),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float]:
        return self.data[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Collate: numpy stack -> pnp.tensor, squeeze batch dim when batch_size == 1
# ---------------------------------------------------------------------------

def _make_collate(batch_size: int):
    squeeze = batch_size == 1

    def collate_fn(batch):
        # batch is a list of (array[P,3], scalar) tuples
        data = np.stack([item[0] for item in batch])       # (B, P, 3)
        labels = np.array([item[1] for item in batch])     # (B,)
        if squeeze:
            data = data[0]      # (P, 3)
            labels = labels[0]  # scalar
        return (
            pnp.tensor(data, requires_grad=False),
            pnp.tensor(labels, requires_grad=False),
        )

    return collate_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataloader(
    config,
    scenario: str = "train",
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Build and return DataLoader(s) for the requested scenario.

    Args:
        config:   OmegaConf / dict-like config object; reads from config.data.*
        scenario: 'train' -> returns (train_loader, val_loader)
                  'test'  -> returns test_loader

    Returns:
        train and val DataLoaders for scenario='train', else a single test DataLoader.
    """
    cfg = config.data
    base_path: str = cfg.base_path
    signal: str = cfg.signal
    background: str = cfg.background
    flatten: bool = bool(cfg.flatten)
    batch_size: int = int(cfg.batch_size)
    num_particles: int = int(cfg.num_particles)

    # n per class: split the total count evenly between signal and background
    n_per_class = {
        "train": int(cfg.n_train) // 2,
        "val":   int(cfg.n_val)   // 2,
        "test":  int(cfg.n_test)  // 2,
    }

    seed: int = int(config.general.seed)
    collate_fn = _make_collate(batch_size)

    def _build(split: str) -> DataLoader:
        sig_files = _get_sorted_files(base_path, split, signal, flatten)
        bkg_files = _get_sorted_files(base_path, split, background, flatten)
        n = n_per_class[split]
        sig_data = _load_jets(sig_files, n, num_particles)
        bkg_data = _load_jets(bkg_files, n, num_particles)
        dataset = JetDataset(sig_data, bkg_data)
        generator = torch.Generator().manual_seed(seed) if split == "train" else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            generator=generator,
            collate_fn=collate_fn,
            num_workers=0,   # h5py is not fork-safe; keep single-process
            pin_memory=False,
        )

    if scenario == "test":
        return _build("test")

    return _build("train"), _build("val")
