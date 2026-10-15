"""
Unit tests for ChakrSpanGroundingModel and SpanGroundingHead.
Verifies multi-token span extraction, positional disambiguation, and boundary indexing.
"""
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.span_grounding import ChakrSpanGroundingModel, ExtractedSpan


def test_span_grounding_forward():
    cfg = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=2, max_seq_len=64)
    model = ChakrSpanGroundingModel(cfg, max_slots=4)

    input_ids = torch.randint(0, 100, (2, 20))
    prompt_len = 12

    vocab_logits, start_logits, end_logits, type_logits, conf = model(input_ids, prompt_len, slot_idx=0)

    assert vocab_logits.shape == (2, 20, 100)
    assert start_logits.shape == (2, 12)
    assert end_logits.shape == (2, 12)
    assert type_logits.shape == (2, 4)
    assert conf.shape == (2, 1)


def test_span_extraction_boundaries():
    cfg = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=2, max_seq_len=64)
    model = ChakrSpanGroundingModel(cfg, max_slots=4)

    input_ids = torch.randint(0, 100, (1, 15))
    prompt_len = 10

    span = model.predict_span(input_ids, prompt_len, slot_idx=1)
    assert isinstance(span, ExtractedSpan)
    assert 0 <= span.start_idx <= span.end_idx < prompt_len
    assert span.span_type in ["NONE", "NUMBER", "STRING", "IDENTIFIER"]
    assert 0.0 <= span.confidence <= 1.0


def test_multi_token_span_preservation():
    cfg = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=2, max_seq_len=64)
    model = ChakrSpanGroundingModel(cfg, max_slots=4)

    # Fake tokenizer mock
    class DummyTokenizer:
        def decode(self, ids):
            return "-".join(str(i) for i in ids)

    input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], dtype=torch.long)
    prompt_len = 6

    # Test span extraction with dummy tokenizer
    span = model.predict_span(input_ids, prompt_len, slot_idx=0, tokenizer=DummyTokenizer())
    assert span.start_idx <= span.end_idx
    assert len(span.span_text) > 0
