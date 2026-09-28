"""
Interactive Cognitive Inference Engine for ChakrView (Step 10).

Provides production-grade autoregressive generation substrate:
- Persistent Key-Value (KV) cache for O(N) single-token incremental decoding
- Token-by-token streaming generator
- Multi-strategy sampling (Greedy, Temperature, Top-K, Top-P, Repetition Penalty)
- Context boundary enforcement (T <= 512, overflow policies)
- Hardware-plan integration (CPU multi-threading, memory limits)
- Numerical safety (finite logit checks, bounds validation)
- Privacy-first structured observability (zero prompt persistence)
- Strict security boundary (pure compute, zero OS/shell authority)
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import time
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union, Any
import torch

from chakrview.brain.cache import KVCache, KVCacheOverflowError
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.hardware import ModelExecutionPlan, PrecisionType
from chakrview.runtime.sampling import SamplingConfig, Sampler
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.runtime.skills import (
    Skill,
    SkillPolicy,
    SkillRegistry,
    SkillDomain,
    SkillResolver,
    RuleBasedSkillResolver,
    get_standard_skill_registry,
)
from chakrview.runtime.knowledge import (
    KnowledgeIndex,
    Retriever,
    LexicalRetriever,
    KnowledgeChunk,
    KnowledgeProvenance,
)
from chakrview.runtime.context import (
    PromptContextBuilder,
    ContextBudget,
    AssembledContext,
)
from chakrview.runtime.tools import (
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    get_standard_tool_registry,
)
from chakrview.runtime.memory import (
    ConversationStore,
    ConversationState,
    ConversationTurn,
    MemoryType,
    MemoryItem,
    WorkingMemory,
    MemoryExtractor,
    ConversationSummarizer,
)


class StopReason(str, Enum):
    """Reason for termination of autoregressive generation."""
    EOS = "eos"
    MAX_TOKENS = "max_tokens"
    CONTEXT_LIMIT = "context_limit"
    USER_STOP = "user_stop"
    ERROR = "error"


@dataclass
class StreamToken:
    """
    An incremental token emitted during streaming autoregressive generation.
    
    Attributes:
        token_id: Integer vocabulary token ID.
        text: Lossless UTF-8 decoded text fragment for this token.
        cumulative_token_count: Number of generated tokens emitted so far.
        is_final: True if this token concludes the sequence.
        stop_reason: Termination reason if is_final is True.
    """
    token_id: int
    text: str
    cumulative_token_count: int
    is_final: bool = False
    stop_reason: Optional[StopReason] = None


@dataclass
class GenerationConfig:
    """
    Runtime controls for autoregressive generation.
    
    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        sampling: Sampling strategy parameters.
        stop_token_ids: List of token IDs that trigger sequence termination (default: [1] for EOS).
        context_window: Hard upper bound on sequence length (<= 512).
        context_overflow_policy: Policy when prompt + generation reaches context_window
                                 ("stop", "truncate", "sliding_window").
        stream_interval: Emit stream event every N tokens.
    """
    max_new_tokens: int = 64
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    stop_token_ids: List[int] = field(default_factory=lambda: [1])  # EOS = 1
    context_window: int = 512
    context_overflow_policy: str = "stop"
    stream_interval: int = 1

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if not (1 <= self.context_window <= 512):
            raise ValueError(f"context_window must be in [1, 512], got {self.context_window}")
        if self.context_overflow_policy not in ("stop", "truncate", "sliding_window"):
            raise ValueError(
                f"Invalid context_overflow_policy: '{self.context_overflow_policy}'. "
                "Must be 'stop', 'truncate', or 'sliding_window'."
            )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["sampling"] = self.sampling.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        data = dict(data)
        if "sampling" in data and isinstance(data["sampling"], dict):
            data["sampling"] = SamplingConfig.from_dict(data["sampling"])
        return cls(**data)


@dataclass
class InferenceMetrics:
    """
    Structured performance, hardware, and latency accounting for an inference run.
    
    Privacy-first: Does NOT persist prompt text.
    """
    model_version: str
    tokenizer_checksum: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    prefill_latency_ms: float
    decode_latency_ms: float
    total_latency_ms: float
    throughput_tokens_per_sec: float
    first_token_latency_ms: float
    stop_reason: StopReason
    cache_memory_bytes: int
    device: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["stop_reason"] = self.stop_reason.value
        return data


@dataclass
class GenerationResult:
    """Complete output package from an inference generation."""
    text: str
    token_ids: List[int]
    metrics: InferenceMetrics
    stop_reason: StopReason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "token_ids": self.token_ids,
            "metrics": self.metrics.to_dict(),
            "stop_reason": self.stop_reason.value,
        }


@dataclass
class RAGResponse:
    """
    Structured outcome of a governed RAG + Skill execution.
    
    Attributes:
        text: Final generated completion text.
        token_ids: Discrete integer tokens generated.
        stop_reason: Generation termination reason (EOS, MAX_TOKENS, CONTEXT_LIMIT).
        metrics: Latency, throughput, and hardware accounting.
        skill_id: Active capability skill identifier used.
        skill_domain: Active capability domain (coding, mathematics, enterprise, etc.).
        knowledge_used: Whether external knowledge chunks were retrieved and injected.
        sources: Fine-grained provenance records for all cited knowledge chunks.
        tool_calls: Structured outcomes of any governed tools invoked during resolution.
        prompt_tokens_count: Number of prompt tokens fed to the model.
        prompt_context: Bounded prompt string used for inference.
    """
    text: str
    token_ids: List[int]
    stop_reason: StopReason
    metrics: InferenceMetrics
    skill_id: Optional[str] = None
    skill_domain: Optional[str] = None
    knowledge_used: bool = False
    sources: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    prompt_tokens_count: int = 0
    prompt_context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "token_ids": self.token_ids,
            "stop_reason": self.stop_reason.value,
            "metrics": self.metrics.to_dict(),
            "skill_id": self.skill_id,
            "skill_domain": self.skill_domain,
            "knowledge_used": self.knowledge_used,
            "sources": self.sources,
            "tool_calls": self.tool_calls,
            "prompt_tokens_count": self.prompt_tokens_count,
            "prompt_context": self.prompt_context,
        }


@dataclass
class ChatResponse:
    """
    Structured outcome of a multi-turn conversation turn in ChakrView (Step 12).
    
    Attributes:
        text: Final generated completion text for this turn.
        session_id: Conversation session identifier.
        turn_id: Unique identifier for this assistant turn.
        token_ids: Discrete integer tokens generated.
        stop_reason: Generation termination reason (EOS, MAX_TOKENS, CONTEXT_LIMIT).
        metrics: Latency, throughput, and hardware accounting.
        skill_id: Active capability skill identifier used.
        skill_domain: Active capability domain.
        working_memories_used: Working memory items surfaced into context.
        knowledge_used: Whether external knowledge was retrieved and injected.
        sources: Citations for knowledge chunks used.
        tool_calls: Structured outcomes of any governed tools invoked.
        turns_in_context: Number of previous conversation turns included in context.
        prompt_tokens_count: Number of prompt tokens fed to the model.
        prompt_context: Bounded prompt string used for inference.
    """
    text: str
    session_id: str
    turn_id: str
    token_ids: List[int]
    stop_reason: StopReason
    metrics: InferenceMetrics
    skill_id: Optional[str] = None
    skill_domain: Optional[str] = None
    working_memories_used: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_used: bool = False
    sources: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    turns_in_context: int = 0
    prompt_tokens_count: int = 0
    prompt_context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "token_ids": self.token_ids,
            "stop_reason": self.stop_reason.value,
            "metrics": self.metrics.to_dict(),
            "skill_id": self.skill_id,
            "skill_domain": self.skill_domain,
            "working_memories_used": self.working_memories_used,
            "knowledge_used": self.knowledge_used,
            "sources": self.sources,
            "tool_calls": self.tool_calls,
            "turns_in_context": self.turns_in_context,
            "prompt_tokens_count": self.prompt_tokens_count,
            "prompt_context": self.prompt_context,
        }


class InferenceSession:
    """
    Interactive, stateful inference session coordinating model, tokenizer, and KV cache.
    
    Architectural Boundaries:
    - Pure computation engine: zero OS/filesystem/shell authority
    - Thread-safe within a single execution session
    - Reusable across applications (chat, tools, coding, reasoning, agents)
    """

    def __init__(
        self,
        model: ChakrMicro,
        tokenizer: BPETokenizer,
        execution_plan: Optional[ModelExecutionPlan] = None,
        sampler: Optional[Sampler] = None,
        model_version: str = "chakrmicro-v0.1",
        tokenizer_checksum: str = "7498d92adeef7c6db98d89a444a7f0e303dd5e7ea4b679a95781a95e6347c617",
        conversation_store: Optional[ConversationStore] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.execution_plan = execution_plan
        self.sampler = sampler or Sampler()
        self.model_version = model_version
        self.tokenizer_checksum = tokenizer_checksum

        # Set evaluation mode
        self.model.eval()

        # Determine target device and context ceiling
        self.device = torch.device(execution_plan.device) if execution_plan else torch.device("cpu")
        self.max_context = execution_plan.max_context_len if execution_plan else self.model.config.max_seq_len
        
        # Configure PyTorch CPU threading if specified
        if execution_plan and execution_plan.device == "cpu" and execution_plan.thread_count > 0:
            torch.set_num_threads(execution_plan.thread_count)

        # Initialize KV Cache
        self.kv_cache: KVCache = KVCache(
            num_layers=self.model.config.n_layers,
            max_seq_len=self.max_context,
            device=self.device,
            dtype=torch.float32,
        )

        self._active_prompt_ids: List[int] = []
        self._generated_token_ids: List[int] = []
        self._latest_logits: Optional[torch.Tensor] = None
        self.conversation_store: ConversationStore = conversation_store or ConversationStore()

    def reset(self) -> None:
        """Reset KV cache and session state for next generation."""
        self.kv_cache.reset()
        self._active_prompt_ids.clear()
        self._generated_token_ids.clear()
        self._latest_logits = None

    def get_conversation(self, session_id: str) -> Optional[ConversationState]:
        """Retrieve conversation state for a session."""
        return self.conversation_store.get_session(session_id)

    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation turns and working memory for a session."""
        return self.conversation_store.clear_session(session_id)

    def delete_conversation(self, session_id: str) -> bool:
        """Delete an entire conversation session."""
        return self.conversation_store.delete_session(session_id)

    def prefill(self, prompt: str, add_bos: bool = True) -> Tuple[List[int], float]:
        """
        Encode prompt and execute initial parallel prefill through ChakrMicro.
        
        Args:
            prompt: Input text sequence.
            add_bos: Whether to prepend BOS token (default: True).
            
        Returns:
            Tuple of (prompt_token_ids, prefill_latency_ms).
        """
        self.reset()
        t0 = time.perf_counter()

        # 1. Encode prompt
        token_ids = self.tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
        if not token_ids:
            # Fallback if empty: inject BOS
            token_ids = [0]

        # 2. Enforce context capacity
        if len(token_ids) >= self.max_context:
            # Prompt exceeds maximum sequence length
            token_ids = token_ids[-self.max_context + 1:]

        self._active_prompt_ids = token_ids
        prompt_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # 3. Model Prefill forward pass
        with torch.no_grad():
            logits, _ = self.model.prefill(prompt_tensor, kv_cache=self.kv_cache)
            # Store last token's logits for subsequent sampling
            self._latest_logits = logits[0, -1, :]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return token_ids, latency_ms

    def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        add_bos: bool = True,
    ) -> Iterator[StreamToken]:
        """
        Stream autoregressively generated tokens one by one.
        
        Args:
            prompt: Input text prompt.
            config: Optional generation configuration override.
            add_bos: Whether to prepend BOS token.
            
        Yields:
            StreamToken for each emitted token until sequence termination.
        """
        cfg = config or GenerationConfig()
        prompt_ids, prefill_latency = self.prefill(prompt, add_bos=add_bos)

        stop_token_set = set(cfg.stop_token_ids)
        cumulative_count = 0

        while cumulative_count < cfg.max_new_tokens:
            current_seq_len = self.kv_cache.sequence_length

            # 1. Check Context Limit
            if current_seq_len >= cfg.context_window or current_seq_len >= self.max_context:
                if cfg.context_overflow_policy == "stop":
                    yield StreamToken(
                        token_id=0,
                        text="",
                        cumulative_token_count=cumulative_count,
                        is_final=True,
                        stop_reason=StopReason.CONTEXT_LIMIT,
                    )
                    return
                elif cfg.context_overflow_policy == "sliding_window":
                    # Evict oldest tokens (sliding window)
                    keep_len = self.max_context // 2
                    self.kv_cache.truncate(keep_len)
                else:  # "truncate"
                    yield StreamToken(
                        token_id=0,
                        text="",
                        cumulative_token_count=cumulative_count,
                        is_final=True,
                        stop_reason=StopReason.CONTEXT_LIMIT,
                    )
                    return

            # 2. Sample next token
            next_token_id = self.sampler.sample(
                logits=self._latest_logits,
                generated_tokens=self._generated_token_ids,
                config=cfg.sampling,
                step=cumulative_count,
            )

            # 3. Decode token to UTF-8 text fragment
            text_fragment = self.tokenizer.decode([next_token_id], errors="replace")
            self._generated_token_ids.append(next_token_id)
            cumulative_count += 1

            # 4. Check for Stop Token (EOS)
            if next_token_id in stop_token_set:
                yield StreamToken(
                    token_id=next_token_id,
                    text=text_fragment,
                    cumulative_token_count=cumulative_count,
                    is_final=True,
                    stop_reason=StopReason.EOS,
                )
                return

            # 5. Check for Max Tokens Reached
            if cumulative_count >= cfg.max_new_tokens:
                yield StreamToken(
                    token_id=next_token_id,
                    text=text_fragment,
                    cumulative_token_count=cumulative_count,
                    is_final=True,
                    stop_reason=StopReason.MAX_TOKENS,
                )
                return

            # Emit normal continuation token
            yield StreamToken(
                token_id=next_token_id,
                text=text_fragment,
                cumulative_token_count=cumulative_count,
                is_final=False,
            )

            # 6. Single-token incremental decode for next step
            next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                step_logits = self.model.decode_next(next_tensor, kv_cache=self.kv_cache)
                self._latest_logits = step_logits[0, -1, :]

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        add_bos: bool = True,
    ) -> GenerationResult:
        """
        Execute full autoregressive generation and return aggregated GenerationResult.
        """
        cfg = config or GenerationConfig()
        t_start = time.perf_counter()

        prompt_ids, prefill_latency_ms = self.prefill(prompt, add_bos=add_bos)
        t_prefill_done = time.perf_counter()

        tokens_generated: List[int] = []
        text_fragments: List[str] = []
        stop_reason = StopReason.MAX_TOKENS
        first_token_latency_ms = 0.0

        for token in self.stream(prompt, config=cfg, add_bos=add_bos):
            if token.cumulative_token_count == 1:
                first_token_latency_ms = (time.perf_counter() - t_start) * 1000.0

            if token.token_id != 0 or not token.is_final:
                tokens_generated.append(token.token_id)
                text_fragments.append(token.text)

            if token.is_final:
                stop_reason = token.stop_reason or StopReason.MAX_TOKENS
                break

        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0
        decode_latency_ms = (t_end - t_prefill_done) * 1000.0

        num_gen = len(tokens_generated)
        throughput = (num_gen / (decode_latency_ms / 1000.0)) if decode_latency_ms > 0 else 0.0

        metrics = InferenceMetrics(
            model_version=self.model_version,
            tokenizer_checksum=self.tokenizer_checksum,
            prompt_tokens=len(prompt_ids),
            generated_tokens=num_gen,
            total_tokens=len(prompt_ids) + num_gen,
            prefill_latency_ms=prefill_latency_ms,
            decode_latency_ms=decode_latency_ms,
            total_latency_ms=total_latency_ms,
            throughput_tokens_per_sec=throughput,
            first_token_latency_ms=first_token_latency_ms,
            stop_reason=stop_reason,
            cache_memory_bytes=self.kv_cache.total_memory_bytes,
            device=str(self.device),
        )

        full_text = self.tokenizer.decode(tokens_generated, errors="replace")
        return GenerationResult(
            text=full_text,
            token_ids=tokens_generated,
            metrics=metrics,
            stop_reason=stop_reason,
        )

    def ask(
        self,
        query: str,
        skill: Optional[Union[str, Skill]] = None,
        knowledge: Optional[Union[KnowledgeIndex, Retriever]] = None,
        config: Optional[GenerationConfig] = None,
        top_k: int = 3,
        auto_resolve_skill: bool = True,
        skill_registry: Optional[SkillRegistry] = None,
        skill_resolver: Optional[SkillResolver] = None,
        tool_registry: Optional[ToolRegistry] = None,
        tool_executor: Optional[ToolExecutor] = None,
        context_builder: Optional[PromptContextBuilder] = None,
    ) -> RAGResponse:
        """
        Execute full RAG + Skill governed cognitive inference cycle.
        
        Pipeline:
        1. Resolve Skill: Matches query to skill policy or uses provided skill.
        2. Governed Tools: If skill authorizes tools and query requests calculation, executes safe tool.
        3. Knowledge Retrieval: Queries lexical/BM25 index for relevant grounded chunks.
        4. Context Assembly: Enforces 512-token ceiling and clear data/instruction boundaries.
        5. Autoregressive Inference: Uses KV cache with skill policy temperature/stops.
        6. Validated Output: Returns RAGResponse with provenance citations and execution metrics.
        """
        # 1. Skill Resolution
        s_registry = skill_registry or get_standard_skill_registry()
        if isinstance(skill, Skill):
            active_skill = skill
        elif isinstance(skill, str):
            active_skill = s_registry.get(skill)
            if active_skill is None:
                raise ValueError(f"Skill '{skill}' not found in registry.")
        elif auto_resolve_skill:
            s_resolver = skill_resolver or RuleBasedSkillResolver()
            active_skill = s_resolver.resolve(query, s_registry)
        else:
            active_skill = s_registry.get("skill_general_v1") or s_registry.list_skills()[0]

        # 2. Governed Tool Execution
        tool_calls: List[Dict[str, Any]] = []
        t_exec = tool_executor or ToolExecutor(tool_registry or get_standard_tool_registry())

        if "calculator" in active_skill.policy.allowed_tools:
            q_strip = query.strip()
            expr_to_calc = None
            calc_match = re.search(r"calculate\s+([0-9\s\+\-\*\/\%\^\(\)\.]+)", q_strip, re.IGNORECASE)
            if calc_match:
                expr_to_calc = calc_match.group(1).strip()
            elif q_strip.lower().startswith("calculate "):
                expr_to_calc = q_strip[10:].strip()
            elif any(c in q_strip for c in "+-*/%^") and not any(
                w in q_strip.lower()
                for w in ["what", "who", "why", "how", "when", "where", "explain"]
            ):
                expr_to_calc = q_strip

            if expr_to_calc:
                calc_res = t_exec.execute(
                    "calculator",
                    {"expression": expr_to_calc},
                    active_skill.policy,
                )
                tool_calls.append(calc_res.to_dict())

        # 3. Knowledge Retrieval
        retrieved_chunks: List[Tuple[KnowledgeChunk, float]] = []
        if knowledge is not None:
            if isinstance(knowledge, Retriever):
                retriever = knowledge
            else:
                retriever = LexicalRetriever(knowledge)
            retrieved_chunks = retriever.retrieve(query, top_k=top_k)

        # 4. Context Assembly
        builder = context_builder or PromptContextBuilder()
        gen_tokens = config.max_new_tokens if config else 64
        budget = ContextBudget(
            max_context=self.max_context,
            generation_budget=gen_tokens,
            max_system_tokens=active_skill.policy.max_context_tokens // 4
            if active_skill.policy.max_context_tokens
            else 64,
        )

        assembled_ctx = builder.build_prompt(
            user_query=query,
            system_prompt=active_skill.system_prompt_template,
            retrieved_chunks=retrieved_chunks,
            budget=budget,
            tokenizer=self.tokenizer,
        )

        # 5. Build GenerationConfig respecting SkillPolicy
        cfg = config or GenerationConfig()
        if config is None or config.sampling.temperature == 0.0:
            cfg.sampling.temperature = active_skill.policy.temperature

        # 6. Execute Generation
        gen_result = self.generate(assembled_ctx.full_prompt, config=cfg, add_bos=True)

        # 7. Package RAGResponse
        sources_provenance = [p.to_dict() for p in assembled_ctx.provenance]
        return RAGResponse(
            text=gen_result.text,
            token_ids=gen_result.token_ids,
            stop_reason=gen_result.stop_reason,
            metrics=gen_result.metrics,
            skill_id=active_skill.skill_id,
            skill_domain=active_skill.domain.value,
            knowledge_used=assembled_ctx.knowledge_used,
            sources=sources_provenance,
            tool_calls=tool_calls,
            prompt_tokens_count=assembled_ctx.estimated_prompt_tokens,
            prompt_context=assembled_ctx.full_prompt,
        )

    def chat(
        self,
        session_id: str,
        user_text: str,
        config: Optional[GenerationConfig] = None,
        skill: Optional[Union[Skill, str]] = None,
        knowledge: Optional[Union[KnowledgeIndex, Retriever]] = None,
        skill_registry: Optional[SkillRegistry] = None,
        skill_resolver: Optional[SkillResolver] = None,
        tool_registry: Optional[ToolRegistry] = None,
        tool_executor: Optional[ToolExecutor] = None,
        context_builder: Optional[PromptContextBuilder] = None,
        top_k_knowledge: int = 2,
        top_k_memory: int = 3,
        auto_resolve_skill: bool = True,
        auto_extract_memory: bool = True,
        auto_summarize: bool = True,
    ) -> ChatResponse:
        """
        Execute a multi-turn conversation turn with short-term working memory (Step 12).
        
        The model itself remains strictly stateless; the runtime manages conversation
        state, working memory scoring, and context budget allocation.
        
        Lifecycle Pipeline:
        1. ConversationStore: Appends user turn to active session.
        2. Skill Resolution: Resolves skill policy and domain for current utterance.
        3. Working Memory Extraction: Identifies constraints, preferences, or task state.
        4. Governed Tools: If skill authorizes tools and query requires it, executes tool safely.
        5. Rolling Summarization: Compresses older dialogue if threshold is exceeded.
        6. Working Memory Selection: Ranks and retrieves relevant working memories.
        7. Knowledge Retrieval: Queries lexical/BM25 index for external knowledge.
        8. Context Assembly: Combines System + Memory + RAG + History + Query <= 512 tokens.
        9. Autoregressive Inference: Generates response via KV cache.
        10. ConversationStore Update: Appends assistant turn to session.
        11. Output Packaging: Returns structured ChatResponse with full telemetry.
        """
        if not user_text or not user_text.strip():
            raise ValueError("user_text cannot be empty.")

        # 1. ConversationStore Session Management & User Turn
        state = self.conversation_store.get_session(session_id)
        if state is None:
            state = self.conversation_store.create_session(session_id=session_id)

        user_clean = user_text.strip()
        builder = context_builder or PromptContextBuilder()
        user_tokens = builder.estimate_tokens(user_clean, self.tokenizer)

        user_turn = self.conversation_store.append_turn(
            session_id=session_id,
            role="user",
            text=user_clean,
            token_estimate=user_tokens,
        )

        # 2. Skill Resolution
        s_registry = skill_registry or get_standard_skill_registry()
        if isinstance(skill, Skill):
            active_skill = skill
        elif isinstance(skill, str):
            active_skill = s_registry.get(skill)
            if active_skill is None:
                raise ValueError(f"Skill '{skill}' not found in registry.")
        elif auto_resolve_skill:
            s_resolver = skill_resolver or RuleBasedSkillResolver()
            active_skill = s_resolver.resolve(user_clean, s_registry)
        else:
            active_skill = s_registry.get("skill_general_v1") or s_registry.list_skills()[0]

        state.active_skill_id = active_skill.skill_id
        state.active_domain = active_skill.domain.value

        # 3. Rule-Based Memory Extraction (Deterministic, non-LLM)
        if auto_extract_memory:
            extracted_items = MemoryExtractor.extract_from_text(
                text=user_clean,
                source_turn_id=user_turn.turn_id,
                sequence_num=state.sequence_counter,
            )
            for item in extracted_items:
                state.working_memory.add(item)

        # 4. Governed Tool Execution
        tool_calls: List[Dict[str, Any]] = []
        t_exec = tool_executor or ToolExecutor(tool_registry or get_standard_tool_registry())

        if "calculator" in active_skill.policy.allowed_tools:
            expr_to_calc = None
            calc_match = re.search(r"calculate\s+([0-9\s\+\-\*\/\%\^\(\)\.]+)", user_clean, re.IGNORECASE)
            if calc_match:
                expr_to_calc = calc_match.group(1).strip()
            elif user_clean.lower().startswith("calculate "):
                expr_to_calc = user_clean[10:].strip()
            elif any(c in user_clean for c in "+-*/%^") and not any(
                w in user_clean.lower()
                for w in ["what", "who", "why", "how", "when", "where", "explain"]
            ):
                expr_to_calc = user_clean

            if expr_to_calc:
                calc_res = t_exec.execute(
                    "calculator",
                    {"expression": expr_to_calc},
                    active_skill.policy,
                )
                tool_calls.append(calc_res.to_dict())
                if calc_res.success:
                    tool_toks = builder.estimate_tokens(str(calc_res.output), self.tokenizer)
                    self.conversation_store.append_turn(
                        session_id=session_id,
                        role="tool",
                        text=f"Calculator result: {calc_res.output}",
                        token_estimate=tool_toks,
                        metadata=calc_res.to_dict(),
                    )

        # 5. Rolling Summarization (Deterministic)
        if auto_summarize:
            ConversationSummarizer.maybe_summarize(
                state=state,
                max_turns_threshold=8,
                keep_recent=4,
            )

        # 6. Working Memory Selection
        retrieved_memories = state.working_memory.retrieve_relevant(
            query=user_clean,
            top_k=top_k_memory,
            active_domain=state.active_domain,
        )

        # 7. Knowledge Retrieval (RAG)
        retrieved_chunks: List[Tuple[KnowledgeChunk, float]] = []
        if knowledge is not None:
            if isinstance(knowledge, Retriever):
                retriever = knowledge
            else:
                retriever = LexicalRetriever(knowledge)
            retrieved_chunks = retriever.retrieve(user_clean, top_k=top_k_knowledge)

        # 8. Context Assembly (Guarantees <= 512 tokens)
        prior_turns = [
            t for t in state.turns
            if t.turn_id != user_turn.turn_id and t.role in ("user", "assistant", "system")
        ]
        gen_tokens = config.max_new_tokens if config else 64
        budget = ContextBudget(
            max_context=self.max_context,
            generation_budget=gen_tokens,
            max_system_tokens=active_skill.policy.max_context_tokens // 4
            if active_skill.policy.max_context_tokens
            else 64,
        )

        assembled_ctx = builder.build_prompt(
            user_query=user_clean,
            system_prompt=active_skill.system_prompt_template,
            retrieved_chunks=retrieved_chunks,
            working_memories=retrieved_memories,
            conversation_turns=prior_turns,
            budget=budget,
            tokenizer=self.tokenizer,
        )

        # 9. Autoregressive Inference
        cfg = config or GenerationConfig()
        if config is None or config.sampling.temperature == 0.0:
            cfg.sampling.temperature = active_skill.policy.temperature

        gen_result = self.generate(assembled_ctx.full_prompt, config=cfg, add_bos=True)

        # 10. ConversationStore Update (Store Assistant Turn)
        asst_tokens = len(gen_result.token_ids)
        asst_turn = self.conversation_store.append_turn(
            session_id=session_id,
            role="assistant",
            text=gen_result.text,
            token_estimate=asst_tokens,
            metadata={"stop_reason": gen_result.stop_reason.value},
        )

        # 11. Return Structured ChatResponse
        sources_provenance = [p.to_dict() for p in assembled_ctx.provenance]
        memories_used_dicts = [m.to_dict() for m in assembled_ctx.memories_used]

        return ChatResponse(
            text=gen_result.text,
            session_id=session_id,
            turn_id=asst_turn.turn_id,
            token_ids=gen_result.token_ids,
            stop_reason=gen_result.stop_reason,
            metrics=gen_result.metrics,
            skill_id=active_skill.skill_id,
            skill_domain=active_skill.domain.value,
            working_memories_used=memories_used_dicts,
            knowledge_used=assembled_ctx.knowledge_used,
            sources=sources_provenance,
            tool_calls=tool_calls,
            turns_in_context=len(assembled_ctx.turns_included),
            prompt_tokens_count=assembled_ctx.estimated_prompt_tokens,
            prompt_context=assembled_ctx.full_prompt,
        )
