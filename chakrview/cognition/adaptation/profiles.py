"""
Resource Profiles for ChakrView Adaptation (Step 23).

Classifies hardware into:
- LOW_RESOURCE
- STANDARD
- HIGH_RESOURCE

CRITICAL ARCHITECTURAL INVARIANT:
The assigned profile MUST NOT alter:
- Model architecture (parameters = 3,443,136, 6 layers, d_model=192, 6 heads, d_ff=512)
- Tokenizer vocabulary (4096 tokens)
- Context length maximum (512 tokens)
- Special token IDs (BOS=0, EOS=1, PAD=2)
- Authority model (DATA != AUTHORITY, etc.)
- Security boundaries

Resource profiles ONLY determine execution budgets.
"""

from enum import Enum
from typing import Dict, Any

from chakrview.cognition.adaptation.hardware import HardwareProfileSnapshot, UNKNOWN


class ResourceProfile(str, Enum):
    """
    Categorization of system capability tiers.
    """
    LOW_RESOURCE = "LOW_RESOURCE"        # Older PCs, constrained memory (< 4GB RAM, <= 2 cores)
    STANDARD = "STANDARD"                # Standard modern laptops/workstations (4-16GB RAM, 4-8 cores)
    HIGH_RESOURCE = "HIGH_RESOURCE"      # Multi-core workstations / servers (> 16GB RAM, > 8 cores)


class ResourceClassifier:
    """
    Deterministic rule-based mapper from HardwareProfileSnapshot to ResourceProfile.
    """

    @staticmethod
    def classify(snapshot: HardwareProfileSnapshot) -> ResourceProfile:
        """Classify host environment into a bounded ResourceProfile."""
        logical_cores = snapshot.cpu.logical_cores
        avail_bytes = snapshot.memory.available_bytes

        # Check memory bounds if available
        if isinstance(avail_bytes, int):
            avail_gb = avail_bytes / (1024 ** 3)
            if avail_gb < 2.5 or logical_cores <= 2:
                return ResourceProfile.LOW_RESOURCE
            if avail_gb >= 12.0 and logical_cores >= 8:
                return ResourceProfile.HIGH_RESOURCE
            return ResourceProfile.STANDARD

        # Fallback when memory detection is UNKNOWN
        if logical_cores <= 2:
            return ResourceProfile.LOW_RESOURCE
        elif logical_cores >= 8:
            return ResourceProfile.HIGH_RESOURCE
        else:
            return ResourceProfile.STANDARD
