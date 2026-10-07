"""Step 410: Model Registry with Enforced Promotion Transitions.

Implements a strict model lifecycle registry with the state machine:

    EXPERIMENTAL --> FROZEN --> RELEASED --> DEPRECATED

Rules
-----
- Only EXPERIMENTAL candidates may be promoted.
- Promotion from EXPERIMENTAL to FROZEN requires:
    (a) A valid I4CandidateManifest with all_gates_passed=True
    (b) Baseline SHA intact (no drift from canonical)
    (c) Manifest hash verified (tamper detection)
- Promotion from FROZEN to RELEASED requires explicit human approval flag.
- No silent promotions: every state change is recorded in the audit log.
- The canonical baseline (ChakrMicro SHA c5571c...) is NEVER modified.
"""

from __future__ import annotations

import dataclasses
import enum
import time
from typing import Any, Dict, List, Optional, Tuple

from chakrview.runtime.interactive import (
    compute_model_hash,
    instantiate_frozen_baseline,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.i4_candidate_manifest import (
    I4CandidateManifest,
    _compute_manifest_hash,
)


# ---------------------------------------------------------------------------
# Lifecycle enum
# ---------------------------------------------------------------------------

class ModelLifecycleState(str, enum.Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    FROZEN = "FROZEN"
    RELEASED = "RELEASED"
    DEPRECATED = "DEPRECATED"


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ModelRegistryEntry:
    """A single entry in the model registry."""
    candidate_id: str
    state: ModelLifecycleState
    manifest: Optional[I4CandidateManifest]
    audit_log: List[str]
    created_at: str
    last_updated_at: str

    def _log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.audit_log.append(f"[{ts}] {msg}")
        self.last_updated_at = ts


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Singleton-like model registry that enforces lifecycle transitions.

    Usage
    -----
        registry = ModelRegistry()
        entry = registry.register_experimental(candidate_id="my-exp-001")
        ok, reason = registry.promote_to_frozen(candidate_id="my-exp-001", manifest=manifest)
        ok, reason = registry.promote_to_released(candidate_id="my-exp-001", human_approved=True)
    """

    def __init__(self) -> None:
        self._entries: Dict[str, ModelRegistryEntry] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_experimental(self, candidate_id: str) -> ModelRegistryEntry:
        """Creates a new EXPERIMENTAL entry. Raises if already registered."""
        if candidate_id in self._entries:
            raise ValueError(
                f"Candidate '{candidate_id}' is already registered with state "
                f"'{self._entries[candidate_id].state}'."
            )
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry = ModelRegistryEntry(
            candidate_id=candidate_id,
            state=ModelLifecycleState.EXPERIMENTAL,
            manifest=None,
            audit_log=[],
            created_at=ts,
            last_updated_at=ts,
        )
        entry._log(f"Registered as EXPERIMENTAL.")
        self._entries[candidate_id] = entry
        return entry

    # ------------------------------------------------------------------
    # Promotion gates
    # ------------------------------------------------------------------

    def promote_to_frozen(
        self,
        candidate_id: str,
        manifest: I4CandidateManifest,
    ) -> Tuple[bool, str]:
        """
        Promotes EXPERIMENTAL -> FROZEN.

        Checks
        ------
        1. Entry must exist and be in EXPERIMENTAL state.
        2. manifest.all_gates_passed must be True.
        3. Baseline SHA must still match EXPECTED_WEIGHT_HASH.
        4. Manifest hash must be internally consistent.
        """
        if candidate_id not in self._entries:
            return False, f"Candidate '{candidate_id}' not found in registry."

        entry = self._entries[candidate_id]

        if entry.state != ModelLifecycleState.EXPERIMENTAL:
            return False, (
                f"Invalid transition: state is '{entry.state}', "
                f"must be EXPERIMENTAL to promote to FROZEN."
            )

        # Gate 1: all evaluation criteria passed
        if not manifest.all_gates_passed:
            reason = (
                f"Promotion denied: manifest gates not satisfied "
                f"(G4={manifest.multi_seed_g4_mean:.2%}, "
                f"H2={manifest.mean_h2_routing:.2%}, "
                f"LangRet={manifest.language_retention:.4f}, "
                f"ContamZero={manifest.contamination_zero}, "
                f"Freeze={manifest.freeze_status})."
            )
            entry._log(f"FREEZE DENIED: {reason}")
            return False, reason

        # Gate 2: baseline integrity
        baseline = instantiate_frozen_baseline()
        live_sha = compute_model_hash(baseline)
        if live_sha != EXPECTED_WEIGHT_HASH:
            reason = (
                f"Promotion denied: baseline SHA drift detected "
                f"(live={live_sha[:16]}..., expected={EXPECTED_WEIGHT_HASH[:16]}...)."
            )
            entry._log(f"FREEZE DENIED: {reason}")
            return False, reason

        # Gate 3: manifest hash verification
        recomputed = _compute_manifest_hash(manifest.to_dict())
        if recomputed != manifest.manifest_hash:
            reason = (
                "Promotion denied: manifest hash mismatch (possible tampering)."
            )
            entry._log(f"FREEZE DENIED: {reason}")
            return False, reason

        # All gates passed
        entry.manifest = manifest
        entry.state = ModelLifecycleState.FROZEN
        entry._log(
            f"Promoted to FROZEN. manifest_id={manifest.candidate_id}, "
            f"G4={manifest.multi_seed_g4_mean:.2%}, "
            f"H2={manifest.mean_h2_routing:.2%}, "
            f"baseline_sha={manifest.baseline_sha[:16]}..."
        )
        return True, "Successfully promoted to FROZEN."

    def promote_to_released(
        self,
        candidate_id: str,
        human_approved: bool = False,
    ) -> Tuple[bool, str]:
        """
        Promotes FROZEN -> RELEASED.

        Requires explicit human_approved=True (no silent release).
        """
        if candidate_id not in self._entries:
            return False, f"Candidate '{candidate_id}' not found in registry."

        entry = self._entries[candidate_id]

        if entry.state != ModelLifecycleState.FROZEN:
            return False, (
                f"Invalid transition: state is '{entry.state}', "
                f"must be FROZEN to promote to RELEASED."
            )

        if not human_approved:
            reason = "Promotion to RELEASED requires explicit human_approved=True."
            entry._log(f"RELEASE DENIED: {reason}")
            return False, reason

        entry.state = ModelLifecycleState.RELEASED
        entry._log("Promoted to RELEASED with human approval.")
        return True, "Successfully promoted to RELEASED."

    def deprecate(self, candidate_id: str, reason: str = "") -> Tuple[bool, str]:
        """Marks a RELEASED or FROZEN entry as DEPRECATED."""
        if candidate_id not in self._entries:
            return False, f"Candidate '{candidate_id}' not found."
        entry = self._entries[candidate_id]
        if entry.state not in (ModelLifecycleState.FROZEN, ModelLifecycleState.RELEASED):
            return False, f"Only FROZEN or RELEASED entries may be deprecated."
        entry.state = ModelLifecycleState.DEPRECATED
        entry._log(f"Deprecated. Reason: {reason or 'unspecified'}")
        return True, "Deprecated."

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_entry(self, candidate_id: str) -> Optional[ModelRegistryEntry]:
        return self._entries.get(candidate_id)

    def list_entries(self) -> List[ModelRegistryEntry]:
        return list(self._entries.values())

    def get_frozen_candidates(self) -> List[ModelRegistryEntry]:
        return [e for e in self._entries.values() if e.state == ModelLifecycleState.FROZEN]

    def get_released_candidates(self) -> List[ModelRegistryEntry]:
        return [e for e in self._entries.values() if e.state == ModelLifecycleState.RELEASED]


# ---------------------------------------------------------------------------
# Step 410 Report
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Step410RegistryReport:
    registry: ModelRegistry
    promotion_to_frozen_ok: bool
    promotion_to_frozen_reason: str
    promotion_to_released_ok: bool
    promotion_to_released_reason: bool
    invalid_promotion_blocked: bool
    audit_log_entries: int
    summary: str


def run_step410_model_registry(
    manifest: Optional[I4CandidateManifest] = None,
) -> Step410RegistryReport:
    """
    Executes Step 410: builds a registry, registers the I4 candidate,
    promotes it through the lifecycle, and validates that invalid
    transitions are correctly blocked.
    """
    from chakrview.cognition.i4_candidate_manifest import create_i4_candidate_manifest

    if manifest is None:
        manifest = create_i4_candidate_manifest()

    registry = ModelRegistry()

    # Register as experimental
    registry.register_experimental(candidate_id=manifest.candidate_id)

    # Attempt invalid promotion directly to RELEASED (should fail)
    bad_ok, bad_reason = registry.promote_to_released(
        candidate_id=manifest.candidate_id, human_approved=True
    )
    invalid_blocked = not bad_ok  # We expect this to fail

    # Promote to FROZEN
    frozen_ok, frozen_reason = registry.promote_to_frozen(
        candidate_id=manifest.candidate_id, manifest=manifest
    )

    # Attempt RELEASED without human_approved (should fail)
    no_approval_ok, _ = registry.promote_to_released(
        candidate_id=manifest.candidate_id, human_approved=False
    )
    invalid_blocked = invalid_blocked and (not no_approval_ok)

    # Promote to RELEASED with human approval
    released_ok, released_reason = registry.promote_to_released(
        candidate_id=manifest.candidate_id, human_approved=True
    )

    entry = registry.get_entry(manifest.candidate_id)
    audit_count = len(entry.audit_log) if entry else 0

    summary = (
        f"Step 410 | Registry lifecycle: "
        f"EXPERIMENTAL->FROZEN={frozen_ok} | "
        f"FROZEN->RELEASED={released_ok} | "
        f"InvalidPromotionBlocked={invalid_blocked} | "
        f"AuditLog={audit_count} entries | "
        f"FinalState={entry.state.value if entry else 'UNKNOWN'}"
    )

    return Step410RegistryReport(
        registry=registry,
        promotion_to_frozen_ok=frozen_ok,
        promotion_to_frozen_reason=frozen_reason,
        promotion_to_released_ok=released_ok,
        promotion_to_released_reason=released_reason,
        invalid_promotion_blocked=invalid_blocked,
        audit_log_entries=audit_count,
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 410: Running Model Registry Lifecycle Test...")
    rep = run_step410_model_registry()
    print(rep.summary)
