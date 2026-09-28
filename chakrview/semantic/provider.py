"""
EmbeddingProvider adapter for the Sovereign Semantic Encoder (Step 14).

Exposes the trained or initialized SemanticEncoder through the standard
Step 13 EmbeddingProvider interface for seamless integration with
HybridRetriever, InMemoryVectorIndex, and UnifiedRetriever.
"""

from typing import List, Optional, Any, Union
import torch

from chakrview.runtime.retrieval import EmbeddingProvider
from chakrview.semantic.config import SemanticEncoderConfig
from chakrview.semantic.encoder import SemanticEncoder


class NeuralSemanticEmbeddingProvider(EmbeddingProvider):
    """
    Adapter integrating the neural SemanticEncoder into ChakrView's retrieval subsystem.
    
    100% compliant with the Step 13 EmbeddingProvider abstract contract.
    Enables drop-in replacement of DeterministicHashEmbeddingProvider with zero
    modifications to HybridRetriever, InMemoryVectorIndex, or InferenceSession.
    """

    def __init__(
        self,
        encoder: SemanticEncoder,
        tokenizer: Any,
        device: Union[str, torch.device] = "cpu",
        provider_name: str = "chakrview_semantic_encoder_v1",
    ) -> None:
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.encoder.to(self.device)
        self.encoder.eval()
        self._provider_name = provider_name

    @property
    def dimension(self) -> int:
        """Dimensionality of the dense semantic vector."""
        return self.encoder.config.embedding_dim

    @property
    def provider_id(self) -> str:
        """Unique provider identifier."""
        return f"{self._provider_name}_d{self.dimension}"

    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single text string into a dense L2-normalized float vector.
        """
        clean = text.strip()
        if not clean:
            return [0.0] * self.dimension

        with torch.no_grad():
            vec = self.encoder.encode_text(
                tokenizer=self.tokenizer,
                text=clean,
                device=self.device,
            )
        return [float(x) for x in vec.cpu().tolist()]

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """
        Batch-encode multiple text strings into dense L2-normalized float vectors.
        """
        if not texts:
            return []

        cleaned = [t.strip() for t in texts]
        non_empty_indices = [i for i, t in enumerate(cleaned) if len(t) > 0]

        if not non_empty_indices:
            return [[0.0] * self.dimension for _ in texts]

        sub_texts = [cleaned[i] for i in non_empty_indices]

        with torch.no_grad():
            vecs = self.encoder.encode_batch(
                tokenizer=self.tokenizer,
                texts=sub_texts,
                device=self.device,
            )
        vecs_list = vecs.cpu().tolist()

        # Reconstruct full list preserving empty entries
        results: List[List[float]] = [[0.0] * self.dimension for _ in texts]
        for orig_idx, vec in zip(non_empty_indices, vecs_list):
            results[orig_idx] = [float(x) for x in vec]

        return results
