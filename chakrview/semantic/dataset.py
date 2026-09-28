"""
Dataset abstractions for the Sovereign Semantic Encoder (Step 14).

Provides reproducible data structures, local JSONL ingestion, deterministic
train/validation splitting, and tokenizing collators.
"""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Any, Iterator, Union
import torch
from torch.utils.data import Dataset


@dataclass
class SemanticPair:
    """
    Single semantic contrastive training or evaluation record.
    
    Attributes:
        query: Anchor query text.
        positive: Ground-truth relevant document text.
        negatives: Optional list of explicit hard negative texts.
        metadata: Optional provenance or tracking metadata.
    """
    query: str
    positive: str
    negatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticPair":
        return cls(**data)


class SemanticDataset(Dataset):
    """
    In-memory list of SemanticPair instances with reproducible splitting and batching.
    """

    def __init__(self, pairs: Optional[List[SemanticPair]] = None) -> None:
        self.pairs: List[SemanticPair] = list(pairs or [])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> SemanticPair:
        return self.pairs[index]

    def add_pair(self, pair: SemanticPair) -> None:
        self.pairs.append(pair)

    def shuffle(self, seed: int = 42) -> "SemanticDataset":
        """Deterministically shuffle pairs in-place."""
        rng = random.Random(seed)
        shuffled = list(self.pairs)
        rng.shuffle(shuffled)
        return SemanticDataset(shuffled)

    def train_val_split(
        self,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple["SemanticDataset", "SemanticDataset"]:
        """Deterministically partition into train and validation sets."""
        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
        shuffled = self.shuffle(seed=seed).pairs
        val_count = max(1, int(len(shuffled) * val_ratio))
        train_pairs = shuffled[val_count:]
        val_pairs = shuffled[:val_count]
        return SemanticDataset(train_pairs), SemanticDataset(val_pairs)

    def save_jsonl(self, path: Union[str, Path]) -> None:
        """Serialize dataset records to a JSONL file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for item in self.pairs:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: Union[str, Path]) -> "SemanticDataset":
        """Load dataset records from a JSONL file, ignoring empty lines."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        pairs = []
        with open(p, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                clean = line.strip()
                if not clean:
                    continue
                try:
                    obj = json.loads(clean)
                    if "query" not in obj or "positive" not in obj:
                        raise ValueError(f"Missing required 'query' or 'positive' field on line {line_num}")
                    pairs.append(SemanticPair.from_dict(obj))
                except Exception as e:
                    raise ValueError(f"Malformed JSONL line {line_num} in {p}: {e}")
        return cls(pairs)


class SemanticCollator:
    """
    Collate and tokenize SemanticPair samples into PyTorch batch tensors.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_seq_len: int = 256,
        pad_token_id: int = 2,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def _pad_tokens(self, token_lists: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pads token ID lists to the batch maximum length."""
        max_len = max(len(ids) for ids in token_lists)
        padded = []
        masks = []
        for ids in token_lists:
            diff = max_len - len(ids)
            padded.append(ids + [self.pad_token_id] * diff)
            masks.append([1] * len(ids) + [0] * diff)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

    def __call__(self, batch: List[SemanticPair]) -> Dict[str, Any]:
        """
        Tokenizes query, positive, and optional negative texts.
        """
        queries = [p.query for p in batch]
        positives = [p.positive for p in batch]

        q_ids = [self.tokenizer.encode(q, add_bos=True, add_eos=True)[:self.max_seq_len] for q in queries]
        p_ids = [self.tokenizer.encode(p, add_bos=True, add_eos=True)[:self.max_seq_len] for p in positives]

        q_tensor, q_mask = self._pad_tokens(q_ids)
        p_tensor, p_mask = self._pad_tokens(p_ids)

        out = {
            "query_ids": q_tensor,
            "query_mask": q_mask,
            "positive_ids": p_tensor,
            "positive_mask": p_mask,
            "queries": queries,
            "positives": positives,
        }

        # Check if negatives exist for all items
        has_negatives = all(len(p.negatives) > 0 for p in batch)
        if has_negatives:
            # Flatten negatives: for each query, take the first negative
            first_negs = [p.negatives[0] for p in batch]
            n_ids = [self.tokenizer.encode(n, add_bos=True, add_eos=True)[:self.max_seq_len] for n in first_negs]
            n_tensor, n_mask = self._pad_tokens(n_ids)
            out["negative_ids"] = n_tensor
            out["negative_mask"] = n_mask
            out["negatives"] = first_negs

        return out


def create_reference_fixture_dataset() -> SemanticDataset:
    """
    Deterministic reference dataset for local testing and pipeline verification.
    
    Contains synthetic, domain-aligned pairs covering technical, architectural,
    and computational concepts without external dependencies.
    """
    pairs = [
        SemanticPair(
            query="transformer attention mechanism",
            positive="Self-attention layers compute scaled dot-product weights between query and key projections.",
            negatives=["Gradient descent updates model parameters with backpropagated errors."],
            metadata={"domain": "neural_architecture"},
        ),
        SemanticPair(
            query="rotary position embeddings formula",
            positive="RoPE applies complex coordinate rotation matrices to query and key vector representations.",
            negatives=["Working memory tracks session-isolated conversational user preferences."],
            metadata={"domain": "positional_embeddings"},
        ),
        SemanticPair(
            query="BM25 lexical retrieval algorithm",
            positive="Okapi BM25 scores document relevance through term frequency saturation and document length normalization.",
            negatives=["SwiGLU feed-forward networks provide non-linear feature activations."],
            metadata={"domain": "information_retrieval"},
        ),
        SemanticPair(
            query="working memory session isolation",
            positive="Conversational store ensures user state is confined to volatile memory without cross-session leakage.",
            negatives=["Pre-RMSNorm stabilizes deep transformer layers using numerical epsilon constants."],
            metadata={"domain": "conversational_memory"},
        ),
        SemanticPair(
            query="context budget ceiling enforcement",
            positive="Prompt context builder truncates auxiliary history to strictly respect the 512 token ceiling.",
            negatives=["Byte-level BPE compresses Unicode text into discrete token identifiers."],
            metadata={"domain": "context_engineering"},
        ),
        SemanticPair(
            query="swiglu activation function",
            positive="SwiGLU computes elementwise product of Swish activated projection and linear gate projection.",
            negatives=["Cosine similarity measures the angle between normalized embedding vectors."],
            metadata={"domain": "neural_architecture"},
        ),
        SemanticPair(
            query="cosine similarity vector metric",
            positive="Normalized dot product between unit vectors bounds geometric similarity within negative one to one.",
            negatives=["Okapi BM25 scores document relevance through term frequency saturation."],
            metadata={"domain": "vector_mathematics"},
        ),
        SemanticPair(
            query="governed tool sandboxing",
            positive="Abstract syntax tree parsing validates arithmetic expressions before calculator tool invocation.",
            negatives=["Transformer self-attention layers compute scaled dot-product weights."],
            metadata={"domain": "runtime_security"},
        ),
    ]
    return SemanticDataset(pairs)
