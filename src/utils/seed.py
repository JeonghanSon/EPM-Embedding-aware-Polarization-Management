# src/utils/seed.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None  # allow import even when torch is not installed


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Reproducibility helper.

    Args:
        seed: Random seed.
        deterministic: If True, enforce deterministic behavior as much as possible
            (may slow down execution).
        benchmark: If True, allow cuDNN autotuning (may reduce reproducibility).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN / CUDA settings
    torch.backends.cudnn.benchmark = bool(benchmark)
    torch.backends.cudnn.deterministic = bool(deterministic)

    # Enforce deterministic algorithms in PyTorch when possible.
    # Some operations may raise an error; ignore if unsupported.
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        # Deterministic cuBLAS workspace setting (may be required for some GEMM ops).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_worker(worker_id: int) -> None:
    """
    Worker init function for PyTorch DataLoader.
    """
    worker_seed = (os.getpid() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    if torch is not None:
        torch.manual_seed(worker_seed)


def make_torch_generator(seed: int):
    """
    Create a torch.Generator for DataLoader(generator=...).
    """
    if torch is None:
        return None
    g = torch.Generator()
    g.manual_seed(seed)
    return g
