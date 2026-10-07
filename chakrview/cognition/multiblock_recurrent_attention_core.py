"""Step 362: Multi-Block Recurrent-Attention Core Model.

Implements MultiBlockRecurrentChakrMicro:
An isolated candidate model where multiple consecutive layers (default: Layer 2 + Layer 3)
are replaced with RecurrentAttentionCoreBlock instances, while all other layers remain frozen.

Architectural Modes:
- Mode A: Independent State per Block (each block starts with its own learnable s0)
- Mode B: Persistent State L2 -> L3 (the recurrent state s1 produced by Layer 2 is passed
          as the initial state s_in to Layer 3)

Features:
- Baseline remains bit-exact (Delta W = 0).
- Bit-exact initialization: Cycle 1 projections in both blocks copied directly from canonical weights.
- gamma_c2 initialized to 0.0 in both blocks, guaranteeing zero initial output disruption.
- Trainable parameter count:
  * Two blocks (freeze_ffn=True): 2 * 217,537 = 435,074 params + 13,057 binding = 448,131 params (<= 500k budget).
  * Optional shared recurrent transition weights (saves ~14k params).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.recurrent_attention_core_block import (
    RecurrentAttentionCoreBlock,
    CoreBlockTrace,
)
from chakrview.cognition.dynamic_contextual_token_binding import (
    DynamicContextualTokenBinding,
)


def compute_module_sha256(module: nn.Module) -> str:
    """Computes deterministic SHA-256 digest of named module parameters."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(module.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


class MultiBlockRecurrentChakrMicro(nn.Module):
    """
    ChakrMicro candidate replacing multiple layers (e.g., Layers [2, 3]) with
    RecurrentAttentionCoreBlock modules.
    """

    def __init__(
        self,
        base_model: ChakrMicro,
        target_layers: Optional[List[int]] = None,
        d_state: int = 48,
        persistent_state: bool = True,
        shared_transition: bool = False,
        freeze_ffn: bool = True,
        enable_contextual_binding: bool = True,
        d_bind: int = 32,
    ) -> None:
        super().__init__()
        self.base_clone = copy.deepcopy(base_model)
        self.config = self.base_clone.config
        self.d_model = self.config.d_model

        if target_layers is None:
            target_layers = [2, 3]
        self.target_layers = sorted(target_layers)
        self.persistent_state = persistent_state
        self.shared_transition = shared_transition

        # Freeze everything in base clone
        for p in self.base_clone.parameters():
            p.requires_grad = False

        # Build replacement blocks
        self.recurrent_blocks = nn.ModuleDict()
        for idx in self.target_layers:
            block = RecurrentAttentionCoreBlock(
                config=self.config,
                d_state=d_state,
                use_shared_kv=True,
                freeze_ffn=freeze_ffn,
            )
            # Copy canonical weights for layer idx to ensure zero-drift init
            with torch.no_grad():
                orig_block = self.base_clone.layers[idx]
                block.norm_1.weight.copy_(orig_block.norm_1.weight)
                block.norm_2.weight.copy_(orig_block.norm_2.weight)
                block.q1_proj.weight.copy_(orig_block.attn.q_proj.weight)
                block.k1_proj.weight.copy_(orig_block.attn.k_proj.weight)
                block.v1_proj.weight.copy_(orig_block.attn.v_proj.weight)
                block.out1_proj.weight.copy_(orig_block.attn.out_proj.weight)
                block.ffn.gate_proj.weight.copy_(orig_block.ffn.gate_proj.weight)
                block.ffn.up_proj.weight.copy_(orig_block.ffn.up_proj.weight)
                block.ffn.down_proj.weight.copy_(orig_block.ffn.down_proj.weight)

            self.recurrent_blocks[str(idx)] = block

        # If shared_transition is enabled, bind GRU parameters across blocks
        if shared_transition and len(self.target_layers) > 1:
            first_key = str(self.target_layers[0])
            first_block = self.recurrent_blocks[first_key]
            for idx in self.target_layers[1:]:
                b = self.recurrent_blocks[str(idx)]
                b.w_z_h = first_block.w_z_h
                b.w_z_s = first_block.w_z_s
                b.w_r_h = first_block.w_r_h
                b.w_r_s = first_block.w_r_s
                b.w_n_h = first_block.w_n_h
                b.w_n_s = first_block.w_n_s

        # Contextual token binding readout
        if enable_contextual_binding:
            self.binding = DynamicContextualTokenBinding(d_model=self.d_model, d_bind=d_bind)
        else:
            self.binding = None

    @property
    def trainable_param_count(self) -> int:
        seen = set()
        count = 0
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                count += p.numel()
        return count

    @property
    def total_param_count(self) -> int:
        seen = set()
        count = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                count += p.numel()
        return count

    def forward_hidden_states(
        self,
        input_ids: torch.Tensor,
        query_pos: Optional[int] = None,
        override_state: Optional[Dict[int, torch.Tensor]] = None,
        zero_state_layers: Optional[List[int]] = None,
        shuffle_state_between_blocks: bool = False,
        reset_state_between_blocks: bool = False,
        bypass_state_layers: Optional[List[int]] = None,
        disable_cycle2_layers: Optional[List[int]] = None,
        replace_block_with_baseline: Optional[List[int]] = None,
        return_traces: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[int, CoreBlockTrace]]:
        """Executes forward pass with multi-block state routing and intervention hooks."""
        B, T = input_ids.shape
        pos = query_pos if query_pos is not None else (T - 1)
        x = self.base_clone.embedding(input_ids)
        states = [x]
        traces: Dict[int, CoreBlockTrace] = {}

        current_recurrent_state = None
        zero_layers = zero_state_layers or []
        bypass_layers = bypass_state_layers or []
        disable_c2_l = disable_cycle2_layers or []
        replace_base_l = replace_block_with_baseline or []
        overrides = override_state or {}

        for idx in range(self.config.n_layers):
            if idx in self.target_layers:
                if idx in replace_base_l:
                    # Use original frozen baseline block
                    x = self.base_clone.layers[idx](x)
                else:
                    block = self.recurrent_blocks[str(idx)]

                    # Determine input state to this block
                    if self.persistent_state and current_recurrent_state is not None:
                        in_state = current_recurrent_state
                        if reset_state_between_blocks:
                            in_state = None
                        elif shuffle_state_between_blocks:
                            p_idx = torch.randperm(in_state.shape[-1])
                            in_state = in_state[:, p_idx]
                    else:
                        in_state = None

                    # Layer-specific overrides
                    override_s = overrides.get(idx, None)
                    if idx in zero_layers:
                        override_s = torch.zeros(B, block.d_state, device=x.device)

                    bypass_s = (idx in bypass_layers)
                    disable_c2 = (idx in disable_c2_l)

                    x, trace = block(
                        x=x,
                        query_pos=pos,
                        prev_state=in_state,
                        override_s1=override_s,
                        bypass_state=bypass_s,
                        disable_c2=disable_c2,
                        return_trace=return_traces,
                    )

                    if trace is not None:
                        traces[idx] = trace
                        # Update carried state for subsequent recurrent blocks
                        current_recurrent_state = trace.s1
            else:
                x = self.base_clone.layers[idx](x)

            states.append(x)

        final_hidden = self.base_clone.final_norm(x)
        return final_hidden, states, traces

    def forward(
        self,
        input_ids: torch.Tensor,
        candidate_ids: Optional[torch.Tensor] = None,
        candidate_positions: Optional[torch.Tensor] = None,
        query_pos: Optional[int] = None,
        override_state: Optional[Dict[int, torch.Tensor]] = None,
        zero_state_layers: Optional[List[int]] = None,
        shuffle_state_between_blocks: bool = False,
        reset_state_between_blocks: bool = False,
        bypass_state_layers: Optional[List[int]] = None,
        disable_cycle2_layers: Optional[List[int]] = None,
        replace_block_with_baseline: Optional[List[int]] = None,
        return_traces: bool = False,
    ) -> Dict[str, Any]:
        """Full forward pass returning vocabulary logits, binding logits, and block traces."""
        B, T = input_ids.shape
        pos = query_pos if query_pos is not None else (T - 1)

        final_hidden, layer_states, traces = self.forward_hidden_states(
            input_ids=input_ids,
            query_pos=pos,
            override_state=override_state,
            zero_state_layers=zero_state_layers,
            shuffle_state_between_blocks=shuffle_state_between_blocks,
            reset_state_between_blocks=reset_state_between_blocks,
            bypass_state_layers=bypass_state_layers,
            disable_cycle2_layers=disable_cycle2_layers,
            replace_block_with_baseline=replace_block_with_baseline,
            return_traces=return_traces,
        )

        vocab_logits = self.base_clone.lm_head(final_hidden)

        out = {
            "vocab_logits": vocab_logits,
            "final_hidden": final_hidden,
            "layer_states": layer_states,
            "traces": traces,
        }

        if self.binding is not None and candidate_ids is not None and candidate_positions is not None:
            B, N = candidate_positions.shape
            cand_states = torch.stack(
                [final_hidden[b, candidate_positions[b]] for b in range(B)],
                dim=0,
            )
            cand_mask = torch.ones((B, N), dtype=torch.bool, device=final_hidden.device)
            val_rep = final_hidden[:, pos : pos + 1, :]
            bind_logits, bind_probs = self.binding(
                retrieved_value_rep=val_rep,
                candidate_token_states=cand_states,
                candidate_mask=cand_mask,
            )
            out["binding_logits"] = bind_logits
            out["candidate_probs"] = bind_probs

        return out
