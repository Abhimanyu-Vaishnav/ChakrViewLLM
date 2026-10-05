"""
ChakrView Step 92: Resource-Adaptive Inference Pipeline.

Connects the Step 91 NeuralInferenceContract with ChakrView's hardware detection
and runtime strategy framework (HardwareCapability, ResourceDetector, RuntimeStrategy).

Enforces:
- LOW_RESOURCE: Smallest safe context (<= 256), sequential token generation, CPU-only fallback, bounded token generation.
- STANDARD: Standard context (<= 512), standard generation budget.
- ACCELERATED: Exploits GPU/accelerator (CUDA/DirectML) if available, expanded generation budget, batched capability if safe.
- CPU-first guarantee: If GPU is not available, operates deterministically on CPU without crashing or requiring accelerators.
- Graceful degradation on constrained RAM / device limits.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from chakrview.runtime.resource import (
    HardwareCapability,
    ResourceDetector,
    ResourcePolicy,
    RuntimeStrategy,
)
from chakrview.runtime.neural_inference_contract import (
    NeuralInferenceContract,
    NeuralInferencePayload,
    NeuralInferenceOutput,
    NeuralInferenceConfig,
    InferenceExecutionMode,
    InferenceStopReason,
)


@dataclass(frozen=True)
class AdaptiveExecutionBudget:
    """
    Hardware-adapted resource budget for an inference execution.
    """
    strategy: RuntimeStrategy
    effective_device: torch.device
    max_context_window: int
    max_generation_tokens: int
    chunk_size: int
    allow_parallel: bool
    hardware_summary: str


class ResourceAdaptiveInferencePipeline:
    """
    Step 92: Resource-Adaptive Inference Pipeline.
    Evaluates hardware capabilities and tunes execution parameters accordingly.
    """

    def __init__(
        self,
        contract: NeuralInferenceContract,
        hardware_capability: Optional[HardwareCapability] = None,
        force_strategy: Optional[RuntimeStrategy] = None,
    ) -> None:
        self.contract = contract
        self.hardware = hardware_capability or ResourceDetector.detect()
        self.force_strategy = force_strategy
        self.budget = self._compute_execution_budget()

    def _compute_execution_budget(self) -> AdaptiveExecutionBudget:
        """
        Derives concrete execution limits from hardware capability.
        """
        strategy = self.force_strategy or ResourceDetector.determine_strategy(self.hardware).strategy

        if strategy == RuntimeStrategy.ACCELERATED:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            max_ctx = 512  # Model ceiling is 512
            max_gen = 128
            chunk_size = 64
            allow_par = True
            summary = f"ACCELERATED on {device} (VRAM: {self.hardware.vram_gb:.1f}GB)"
        elif strategy == RuntimeStrategy.STANDARD:
            device = torch.device("cpu")
            max_ctx = 512
            max_gen = 64
            chunk_size = 32
            allow_par = (self.hardware.cpu_cores >= 4)
            summary = f"STANDARD on CPU ({self.hardware.cpu_cores} cores, RAM: {self.hardware.ram_gb:.1f}GB)"
        else:  # LOW_RESOURCE
            device = torch.device("cpu")
            max_ctx = 256  # Constrained window to minimize memory pressure
            max_gen = 32
            chunk_size = 16
            allow_par = False
            summary = f"LOW_RESOURCE on CPU ({self.hardware.cpu_cores} core, RAM: {self.hardware.ram_gb:.1f}GB)"

        return AdaptiveExecutionBudget(
            strategy=strategy,
            effective_device=device,
            max_context_window=max_ctx,
            max_generation_tokens=max_gen,
            chunk_size=chunk_size,
            allow_parallel=allow_par,
            hardware_summary=summary,
        )

    def adapt_payload(self, payload: NeuralInferencePayload) -> NeuralInferencePayload:
        """
        Adapts inference payload to conform to the hardware-adapted budget.
        """
        user_max_gen = payload.config.max_new_tokens
        bounded_gen = min(user_max_gen, self.budget.max_generation_tokens)

        adapted_cfg = NeuralInferenceConfig(
            max_new_tokens=bounded_gen,
            min_new_tokens=payload.config.min_new_tokens,
            temperature=payload.config.temperature,
            top_p=payload.config.top_p,
            top_k=payload.config.top_k,
            stop_tokens=list(payload.config.stop_tokens),
            add_bos=payload.config.add_bos,
            add_eos=payload.config.add_eos,
            truncate_overflow=True,  # Always safely truncate if exceeding profile window
            mode=InferenceExecutionMode.RESOURCE_AWARE,
        )

        return NeuralInferencePayload(
            prompt=payload.prompt,
            prompt_tokens=payload.prompt_tokens,
            config=adapted_cfg,
            session_id=payload.session_id,
            tenant_id=payload.tenant_id,
            system_directive=payload.system_directive,
            metadata={
                **payload.metadata,
                "strategy": self.budget.strategy.name,
                "max_context_window": self.budget.max_context_window,
            },
        )

    def execute_adaptive(self, payload: NeuralInferencePayload) -> NeuralInferenceOutput:
        """
        Executes inference within the adaptive resource budget.
        """
        # Save contract context ceiling and temporarily clamp to budget
        orig_max_context = self.contract.max_context
        self.contract.max_context = self.budget.max_context_window

        try:
            adapted_payload = self.adapt_payload(payload)
            output = self.contract.generate(adapted_payload)
            # Augment diagnostics with budget metadata
            output.diagnostics["hardware_strategy"] = self.budget.strategy.name
            output.diagnostics["hardware_summary"] = self.budget.hardware_summary
            output.diagnostics["adapted_max_context"] = self.budget.max_context_window
            return output
        finally:
            self.contract.max_context = orig_max_context
