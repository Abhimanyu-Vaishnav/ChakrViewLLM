"""
Training Run Manifest & Reproducibility System for ChakrView (Step 22).

Captures a complete, verifiable record of all training hyperparameters,
model invariants, dataset fingerprints, and execution environment:
Same configuration + Same dataset + Same seed = Reproducible CPU training trajectory.
"""

from dataclasses import dataclass, asdict, field
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Any


@dataclass
class TrainingRunManifest:
    """
    Formal reproducible record of an offline training execution.
    """
    run_id: str
    seed: int
    model_version: str
    param_count: int
    vocab_size: int
    context_length: int
    tokenizer_fingerprint: str
    dataset_fingerprint: str
    dataset_version: str
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_accumulation_steps: int
    gradient_clipping: float
    total_steps: int
    hardware: str = "CPU"
    threads: int = 1
    framework_version: str = ""
    start_timestamp: str = ""
    end_timestamp: str = ""
    checkpoints_saved: List[str] = field(default_factory=list)
    final_train_loss: float = 0.0
    final_val_loss: Optional[float] = None
    final_perplexity: Optional[float] = None
    reproducibility_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reproducibility_hash:
            self.reproducibility_hash = self.compute_reproducibility_hash()

    def compute_reproducibility_hash(self) -> str:
        """Compute deterministic hash over run configuration parameters."""
        hasher = hashlib.sha256()
        key_str = (
            f"seed:{self.seed}:model:{self.model_version}:tok:{self.tokenizer_fingerprint}:"
            f"data:{self.dataset_fingerprint}:opt:{self.optimizer_name}:lr:{self.learning_rate}:"
            f"bs:{self.batch_size}:ga:{self.gradient_accumulation_steps}:gc:{self.gradient_clipping}:"
            f"steps:{self.total_steps}"
        )
        hasher.update(key_str.encode("utf-8"))
        return hasher.hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingRunManifest":
        return cls(**data)

    def save(self, filepath: Path | str) -> Path:
        out_p = Path(filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_p
