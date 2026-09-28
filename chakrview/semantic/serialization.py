"""
Serialization utilities for the Sovereign Semantic Encoder (Step 14).

Handles atomic saving, loading, versioning, and state verification of
SemanticEncoder checkpoints.
"""

from pathlib import Path
from typing import Dict, Tuple, Union, Any, Optional
import torch

from chakrview.semantic.config import SemanticEncoderConfig
from chakrview.semantic.encoder import SemanticEncoder


def save_semantic_encoder(
    model: SemanticEncoder,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save semantic encoder weights, configuration, and metadata to disk.
    
    Args:
        model: Trained SemanticEncoder instance.
        path: Target file path (.pt).
        metadata: Optional training/evaluation metrics or provenance details.
    Returns:
        Path to saved artifact.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format_version": "1.0",
        "model_type": "chakrview_semantic_encoder",
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }

    # Atomic write to temporary file then replace
    tmp_path = p.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(p)
    return p


def load_semantic_encoder(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[SemanticEncoder, SemanticEncoderConfig]:
    """
    Load a saved semantic encoder checkpoint.
    
    Args:
        path: Path to .pt checkpoint file.
        device: PyTorch device ('cpu' or torch.device).
    Returns:
        Tuple of (instantiated SemanticEncoder, SemanticEncoderConfig).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Semantic encoder checkpoint not found at: {p}")

    ckpt = torch.load(p, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "config" not in ckpt or "state_dict" not in ckpt:
        raise ValueError(f"Invalid checkpoint format in: {p}")

    config = SemanticEncoderConfig.from_dict(ckpt["config"])
    model = SemanticEncoder(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, config
