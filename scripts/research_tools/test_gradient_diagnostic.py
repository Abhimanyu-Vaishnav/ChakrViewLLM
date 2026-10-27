import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath("."))
from chakrview.brain.config_15m import ChakrMicro15MConfig
from chakrview.brain.model import ChakrMicro

def test_gradient_diagnostic_tied_embeddings():
    torch.manual_seed(42)
    cfg = ChakrMicro15MConfig()
    model = ChakrMicro(cfg)
    model.train()

    assert model.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr(), "LM head and embedding must share storage"

    T_TARGET = 100
    T_CONTEXT = 200
    T_NEITHER = 300

    input_ids = torch.tensor([[10, T_CONTEXT, T_TARGET]], dtype=torch.long)
    targets = torch.tensor([[-100, T_TARGET, -100]], dtype=torch.long)

    out = model(input_ids)
    logits = out[0] if isinstance(out, tuple) else out

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()

    emb_grad = model.embedding.weight.grad
    assert emb_grad is not None

    norm_target = emb_grad[T_TARGET].norm().item()
    norm_context = emb_grad[T_CONTEXT].norm().item()
    norm_neither = emb_grad[T_NEITHER].norm().item()

    # 1. Target token receives highest gradient
    assert norm_target > 10.0
    # 2. Context-only token receives substantial indirect gradient through embedding & attention
    assert norm_context > 5.0
    # 3. Non-context, non-target token receives tiny softmax background gradient
    assert norm_neither < 0.1
    # 4. Refutation check: Context token gradient is NOT zero!
    assert norm_context > 50.0 * norm_neither
