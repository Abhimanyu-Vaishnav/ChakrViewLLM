import sys
import json
import os
import torch

sys.path.insert(0, '.')
from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig

engine = ChakrViewInferenceEngine17M(
    checkpoint_path='experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
tok = engine.tokenizer
print('Tokenizer type:', type(tok))
print('Vocab size:', getattr(tok, 'vocab_size', len(getattr(tok, 'vocab', {}))))

# Check special tokens
specials = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<user>', '<assistant>', '<system>']
for s in specials:
    tid = tok.encode(s)
    print(f'Encode "{s}": {tid}')

samples = [
    'Hello world!',
    'नमस्ते दुनिया',
    'def add(a, b):\n    return a + b',
    'x = 10; y = 20; z = x + y'
]
for s in samples:
    enc = tok.encode(s)
    dec = tok.decode(enc)
    print(f'\nSample: {repr(s)}')
    print(f'Encoded ({len(enc)} tokens): {enc[:15]}')
    print(f'Decoded: {repr(dec)}')
    print(f'Exact match: {s == dec}')
