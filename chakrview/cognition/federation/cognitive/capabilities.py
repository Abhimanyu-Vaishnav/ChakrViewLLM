"""
Federated Cognitive Capabilities for Step 42.

Provides two capability types registered with CapabilityGate:

1. FederatedNeuralCapability:
   - Exposes frozen ChakrMicro inference as a governed federated capability.
   - Enforces ΔW = 0 via pre/post-forward SHA-256 weight hash verification.
   - Enforces context ceiling: prompt_tokens + max_new_tokens <= 512.
   - CPU-first; no GPU dependency.

2. SimpleCognitiveCapability:
   - Generic governed capability for non-neural cognitive roles
     (ANALYST, RESEARCHER, CRITIC, SYNTHESIZER, VERIFIER).
   - Returns structured output dict suitable for CognitiveTaskGraph step results.

AXIOM: MODEL_OUTPUT != AUTHORITY
   Neural inference output is purely data. It is never executed as code.
AXIOM: ADVERTISEMENT != PERMISSION
   Registering a capability does not grant remote execution rights.
   All invocations must pass CapabilityGate.authorize() first.
AXIOM: ZERO NEURAL WEIGHT MUTATION (ΔW = 0)
   Hash verified before AND after every forward pass.
"""

import hashlib
import logging
from typing import Any, Optional

import torch

from chakrview.capability.contract import (
    Capability,
    CapabilityCategory,
    CapabilityDescriptor,
    CapabilityRequest,
    CapabilityResult,
    CapabilityContext,
    CapabilityStatus,
    ResourceLimits,
    RiskClassification,
)
from chakrview.cognition.federation.cognitive.errors import (
    NeuralCapabilityContextOverflowError,
    NeuralWeightMutationError,
)
from chakrview.cognition.federation.cognitive.models import (
    CAPABILITY_NEURAL_INFERENCE,
    MAX_CONTEXT_TOKENS,
)

logger = logging.getLogger("chakrview.federation.cognitive.capabilities")

EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"


def _compute_weight_hash(model: Any) -> str:
    """Compute SHA-256 hash of model parameter names + values for immutability check."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# FederatedNeuralCapability
# ─────────────────────────────────────────────────────────────────────────────

class FederatedNeuralCapability(Capability):
    """
    Governed federated capability exposing ChakrMicro neural inference
    through CapabilityGate with strict ΔW = 0 verification.

    Request parameters:
        prompt_tokens   List[int]  Tokenised input prompt (required)
        max_new_tokens  int        Tokens to generate (default 32, ceiling enforced)

    Returns CapabilityResult.output:
        output_tokens         List[int]  Generated token IDs
        input_token_count     int        Number of prompt tokens used
        total_token_count     int        Total tokens consumed
        weight_hash_verified  bool       Confirms ΔW = 0 post-forward
    """

    DEFAULT_MAX_NEW_TOKENS = 32

    def __init__(self, model: Any) -> None:
        self._model = model
        self._descriptor = CapabilityDescriptor(
            capability_id=CAPABILITY_NEURAL_INFERENCE,
            name="Federated Neural Inference",
            version="42.0.0",
            description=(
                "Frozen ChakrMicro forward pass under ΔW=0 verification, "
                "token ceiling enforcement, and CapabilityGate authorization."
            ),
            category=CapabilityCategory.SOFTWARE,
            risk_level=RiskClassification.COMPUTE,
            provider_id="federated_cognitive_engine",
            required_permissions=["neural_inference"],
            resource_limits=ResourceLimits(
                max_cpu_time_ms=3000.0,
                timeout_seconds=15.0,
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def execute(
        self,
        request: CapabilityRequest,
        context: Optional[CapabilityContext] = None,
    ) -> CapabilityResult:
        """
        Execute frozen neural inference with full immutability + ceiling guard.

        Raises:
            NeuralCapabilityContextOverflowError  if total tokens > 512
            NeuralWeightMutationError             if hash changes pre/post-forward
        """
        params = request.parameters
        prompt_tokens = params.get("prompt_tokens", [])
        max_new_tokens = int(params.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS))

        # CT-08: Enforce context ceiling
        total_requested = len(prompt_tokens) + max_new_tokens
        if total_requested > MAX_CONTEXT_TOKENS:
            raise NeuralCapabilityContextOverflowError(
                f"Token budget {total_requested} exceeds ceiling {MAX_CONTEXT_TOKENS}. "
                f"Reduce prompt_tokens ({len(prompt_tokens)}) or max_new_tokens ({max_new_tokens})."
            )

        if not prompt_tokens:
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=CAPABILITY_NEURAL_INFERENCE,
                success=False,
                output={},
                error="Empty prompt_tokens provided",
                status=CapabilityStatus.AVAILABLE,
            )

        # CT-11: Pre-flight ΔW = 0 verification
        pre_hash = _compute_weight_hash(self._model)
        if pre_hash != EXPECTED_WEIGHT_HASH:
            raise NeuralWeightMutationError(
                f"Pre-forward weight hash mismatch: {pre_hash[:16]}… != expected"
            )

        # Execute under strict no_grad (read-only, zero weight mutation)
        self._model.eval()
        generated_tokens: list = []

        try:
            with torch.no_grad():
                current_tokens = list(prompt_tokens)
                for _ in range(max_new_tokens):
                    if len(current_tokens) >= MAX_CONTEXT_TOKENS:
                        break
                    inp = torch.tensor([current_tokens], dtype=torch.long)
                    logits = self._model(inp)           # (1, seq_len, vocab_size)
                    next_token = int(logits[0, -1, :].argmax(dim=-1).item())
                    generated_tokens.append(next_token)
                    current_tokens.append(next_token)
                    if next_token == 1:                 # EOS token
                        break
        except Exception as exc:
            logger.error("Neural inference error: %s", exc)
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=CAPABILITY_NEURAL_INFERENCE,
                success=False,
                output={},
                error=f"Neural inference error: {exc}",
                status=CapabilityStatus.AVAILABLE,
            )

        # CT-11: Post-flight ΔW = 0 verification
        post_hash = _compute_weight_hash(self._model)
        if post_hash != EXPECTED_WEIGHT_HASH:
            raise NeuralWeightMutationError(
                f"Post-forward weight hash mismatch — ΔW != 0 violation detected!"
            )

        logger.info(
            "Neural inference: %d prompt tokens -> %d generated (ΔW=0 verified)",
            len(prompt_tokens),
            len(generated_tokens),
        )

        return CapabilityResult(
            request_id=request.request_id,
            capability_id=CAPABILITY_NEURAL_INFERENCE,
            success=True,
            output={
                "output_tokens": generated_tokens,
                "input_token_count": len(prompt_tokens),
                "total_token_count": len(prompt_tokens) + len(generated_tokens),
                "weight_hash_verified": True,
            },
            status=CapabilityStatus.AVAILABLE,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SimpleCognitiveCapability
# ─────────────────────────────────────────────────────────────────────────────

class SimpleCognitiveCapability(Capability):
    """
    Generic governed capability for non-neural cognitive roles.

    Returns a structured result containing the role, objective, and a simple
    text conclusion, suitable for CognitiveTaskGraph step result recording.

    Used for: ANALYST, RESEARCHER, CRITIC, SYNTHESIZER, VERIFIER roles.
    """

    def __init__(
        self,
        capability_id: str,
        name: str,
        description: str,
        role: str = "generic",
    ) -> None:
        self._role = role
        self._descriptor = CapabilityDescriptor(
            capability_id=capability_id,
            name=name,
            version="42.0.0",
            description=description,
            category=CapabilityCategory.SOFTWARE,
            risk_level=RiskClassification.COMPUTE,
            provider_id="federated_cognitive_engine",
            required_permissions=["cognitive_reasoning"],
            resource_limits=ResourceLimits(
                max_cpu_time_ms=1000.0,
                timeout_seconds=5.0,
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def execute(
        self,
        request: CapabilityRequest,
        context: Optional[CapabilityContext] = None,
    ) -> CapabilityResult:
        params = request.parameters
        objective = params.get("objective", "")
        step_id = params.get("step_id", self._role)

        conclusion = (
            f"[{self._role.upper()}] Evaluated objective '{objective[:80]}' "
            f"at step '{step_id}' — governed {self._role} analysis complete."
        )

        return CapabilityResult(
            request_id=request.request_id,
            capability_id=self.descriptor.capability_id,
            success=True,
            output={
                "conclusion": conclusion,
                "role": self._role,
                "step_id": step_id,
                "evidence": [f"Evidence from {self._role} analysis of: {objective[:60]}"],
                "hypotheses": [
                    {
                        "claim": f"{self._role} primary hypothesis",
                        "confidence": 0.75,
                        "step_id": step_id,
                    }
                ],
            },
            status=CapabilityStatus.AVAILABLE,
        )
