"""
ChakrView Local Model Runtime (Step 46).

Unified local execution coordinator providing streaming and batch inference,
stateful multi-turn conversation sessions, and persistent cognitive memory recall
over the verified ChakrMicro neural core.

Invariants Enforced:
1. ΔW = 0: Model weights strictly verified against canonical hash at startup and execution.
2. Parameter Invariant: Exactly 3,443,136 parameters.
3. Vocabulary Invariant: Exactly 4,096 tokens.
4. Context Invariant: Max sequence horizon <= 512 tokens.
5. Secret Scanning: Fails closed on prohibited credential patterns.
6. Tenant Isolation: Distinct sessions cannot cross tenant partitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.inference import (
    GenerationConfig,
    StopReason,
)
from chakrview.runtime.pipeline import (
    InferenceEngine,
    InferenceRequest,
    InferenceResult,
    ModelIdentity,
    NeuralWeightMutationError,
    StreamChunk,
)
from chakrview.runtime.sampling import SamplingConfig
from chakrview.runtime.session import (
    ConversationRole,
    ConversationTurn,
    LocalModelSession,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.tokenizer.tokenizer import BPETokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"
EXPECTED_PARAM_COUNT = 3_443_136
EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"


class LocalModelRuntime:
    """
    Unified local runtime coordinator for ChakrView.
    """

    def __init__(
        self,
        model: ChakrMicro,
        tokenizer: BPETokenizer,
        memory_adapter: Optional[Any] = None,
        expected_weight_hash: str = EXPECTED_WEIGHT_HASH,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.memory_adapter = memory_adapter
        self.expected_weight_hash = expected_weight_hash

        # Initialize core inference engine
        self.engine = InferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            expected_weight_hash=self.expected_weight_hash,
        )

        # Stateful multi-turn conversation sessions: keyed by (tenant_id, session_id)
        self._sessions: Dict[str, LocalModelSession] = {}

        # Pre-flight integrity verification
        self.verify_runtime_integrity()

    @classmethod
    def from_default(
        cls,
        tokenizer_dir: Optional[Path] = None,
        memory_adapter: Optional[Any] = None,
        torch_seed: int = 42,
    ) -> LocalModelRuntime:
        """
        Factory initializing runtime with canonical ChakrMicro weights and tokenizer.
        """
        torch.manual_seed(torch_seed)
        model = ChakrMicro(ModelConfig())
        model.eval()

        tok_path = tokenizer_dir or DEFAULT_TOKENIZER_DIR
        tokenizer, _ = load_tokenizer_artifacts(tok_path)

        return cls(
            model=model,
            tokenizer=tokenizer,
            memory_adapter=memory_adapter,
        )

    def verify_runtime_integrity(self) -> bool:
        """
        Cryptographically verify that model weights match canonical SHA-256 (ΔW = 0).
        """
        cur_hash = self.engine.compute_weight_hash()
        if cur_hash != self.expected_weight_hash:
            raise NeuralWeightMutationError(
                f"Runtime integrity failure: {cur_hash} != {self.expected_weight_hash} (ΔW != 0)."
            )
        assert self.model.count_parameters()["total_parameters"] == EXPECTED_PARAM_COUNT
        return True

    @property
    def model_identity(self) -> ModelIdentity:
        """Access verified model identity metadata."""
        return self.engine.model_identity

    # ─────────────────────────────────────────────────────────────────────────
    # Session Management
    # ─────────────────────────────────────────────────────────────────────────

    def _session_key(self, tenant_id: str, session_id: str) -> str:
        return f"{tenant_id}::{session_id}"

    def get_or_create_session(
        self,
        session_id: str,
        tenant_id: str = "default_tenant",
        system_prompt: Optional[str] = None,
    ) -> LocalModelSession:
        """Retrieve existing session or instantiate a new one."""
        key = self._session_key(tenant_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = LocalModelSession(
                session_id=session_id,
                tenant_id=tenant_id,
                system_prompt=system_prompt,
                max_context=self.model.config.max_seq_len,
                memory_adapter=self.memory_adapter,
            )
        elif system_prompt:
            self._sessions[key].add_system_prompt(system_prompt)
        return self._sessions[key]

    def reset_session(self, session_id: str, tenant_id: str = "default_tenant") -> bool:
        """Reset conversation turns for a given session."""
        key = self._session_key(tenant_id, session_id)
        if key in self._sessions:
            self._sessions[key].clear()
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-Turn Chat APIs
    # ─────────────────────────────────────────────────────────────────────────

    def chat(
        self,
        session_id: str,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        tenant_id: str = "default_tenant",
        system_prompt: Optional[str] = None,
        retrieve_memory: bool = True,
    ) -> InferenceResult:
        """
        Execute a single multi-turn chat round-trip in batch mode.
        """
        session = self.get_or_create_session(
            session_id=session_id,
            tenant_id=tenant_id,
            system_prompt=system_prompt,
        )

        cfg = generation_config or GenerationConfig(max_new_tokens=32)
        assembled_prompt, envelope, _ = session.build_prompt_package(
            user_input=prompt,
            tokenizer=self.tokenizer,
            max_new_tokens=cfg.max_new_tokens,
            retrieve_memory=retrieve_memory,
        )

        req = InferenceRequest(
            prompt=assembled_prompt,
            context_envelope=envelope,
            generation_config=cfg,
            tenant_id=tenant_id,
            session_id=session_id,
            truncate_if_overflow=True,
        )

        res = self.engine.execute(req)

        # Record turns in session history
        user_tok_count = len(self.tokenizer.encode(prompt, add_bos=False, add_eos=False))
        session.add_turn(ConversationRole.USER, prompt, user_tok_count)
        session.add_turn(ConversationRole.ASSISTANT, res.text, res.output_token_count)

        return res

    def stream_chat(
        self,
        session_id: str,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        tenant_id: str = "default_tenant",
        system_prompt: Optional[str] = None,
        retrieve_memory: bool = True,
    ) -> Generator[StreamChunk, None, None]:
        """
        Stream a multi-turn chat round-trip chunk-by-chunk in real-time.
        """
        session = self.get_or_create_session(
            session_id=session_id,
            tenant_id=tenant_id,
            system_prompt=system_prompt,
        )

        cfg = generation_config or GenerationConfig(max_new_tokens=32)
        assembled_prompt, envelope, _ = session.build_prompt_package(
            user_input=prompt,
            tokenizer=self.tokenizer,
            max_new_tokens=cfg.max_new_tokens,
            retrieve_memory=retrieve_memory,
        )

        req = InferenceRequest(
            prompt=assembled_prompt,
            context_envelope=envelope,
            generation_config=cfg,
            tenant_id=tenant_id,
            session_id=session_id,
            truncate_if_overflow=True,
        )

        emitted_tokens: List[str] = []
        token_count = 0

        for chunk in self.engine.stream(req):
            emitted_tokens.append(chunk.token_text)
            token_count += 1
            yield chunk

        # After streaming finishes, commit turn to history
        full_assistant_reply = "".join(emitted_tokens)
        user_tok_count = len(self.tokenizer.encode(prompt, add_bos=False, add_eos=False))
        session.add_turn(ConversationRole.USER, prompt, user_tok_count)
        session.add_turn(ConversationRole.ASSISTANT, full_assistant_reply, token_count)

    # ─────────────────────────────────────────────────────────────────────────
    # Stateless Prompt Generation APIs
    # ─────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
    ) -> InferenceResult:
        """Stateless batch text generation."""
        req = InferenceRequest(
            prompt=prompt,
            generation_config=generation_config,
            tenant_id=tenant_id,
            session_id=session_id,
            truncate_if_overflow=True,
        )
        return self.engine.execute(req)

    def stream_generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
    ) -> Generator[StreamChunk, None, None]:
        """Stateless real-time streaming text generation."""
        req = InferenceRequest(
            prompt=prompt,
            generation_config=generation_config,
            tenant_id=tenant_id,
            session_id=session_id,
            truncate_if_overflow=True,
        )
        yield from self.engine.stream(req)
