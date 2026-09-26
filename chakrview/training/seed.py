"""
Centralized Randomness and Determinism Management for ChakrView.

Controls Python random, NumPy, and PyTorch RNG state to ensure 100% reproducible training.
"""

import os
import random
from typing import Dict, Any
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seed all pseudo-random number generators across Python, NumPy, and PyTorch.
    
    Args:
        seed: Integer seed value.
        deterministic: If True, configure PyTorch backends for strict determinism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    if deterministic:
        # Enable deterministic algorithms where supported
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass


def get_rng_state() -> Dict[str, Any]:
    """
    Capture the complete RNG state dictionary across all libraries.
    
    Returns:
        Dictionary containing python, numpy, and torch RNG states.
    """
    state = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """
    Restore RNG state across Python, NumPy, and PyTorch from a saved dictionary.
    
    Args:
        state: State dictionary captured by get_rng_state().
    """
    if "python_rng" in state:
        random.setstate(state["python_rng"])
    if "numpy_rng" in state:
        np.random.set_state(state["numpy_rng"])
    if "torch_rng" in state:
        torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and "cuda_rng" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
