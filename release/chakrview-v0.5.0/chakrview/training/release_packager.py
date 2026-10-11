"""
Release Packager for ChakrMicro Model Releases (Phase 7).

Bundles:
1. Trained checkpoint (`model.pt` and `latest_checkpoint.json`)
2. Tokenizer artifacts (`vocab.json`, `merges.json`, `config.json`)
3. Model and training configuration files
4. Release evaluation metrics report (`eval_summary.json`)
5. Checksums file (`checksums.sha256`)
6. Model card (`MODEL_CARD.md`) and release documentation
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Union

from chakrview.training.release_evaluator import ReleaseEvaluationMetrics


def compute_file_sha256(path: Union[str, Path]) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ReleasePackager:
    """Packages a trained ChakrMicro release bundle."""

    def __init__(self, output_dir: Union[str, Path]) -> None:
        self.output_dir = Path(output_dir)

    def package_release(
        self,
        checkpoint_path: Union[str, Path],
        tokenizer_dir: Union[str, Path],
        eval_metrics: ReleaseEvaluationMetrics,
        model_config_dict: Dict[str, Any],
        training_config_dict: Dict[str, Any],
        release_tag: str = "chakrmicro_v0.1",
    ) -> Path:
        target_dir = self.output_dir / release_tag
        target_dir.mkdir(parents=True, exist_ok=True)

        ckpt_dir = target_dir / "checkpoint"
        tok_dir = target_dir / "tokenizer"
        cfg_dir = target_dir / "config"
        eval_dir = target_dir / "evaluation"

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tok_dir.mkdir(parents=True, exist_ok=True)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy checkpoint
        src_ckpt = Path(checkpoint_path)
        dest_ckpt = ckpt_dir / "model.pt"
        shutil.copy2(src_ckpt, dest_ckpt)

        latest_meta = {
            "release_tag": release_tag,
            "weight_hash": eval_metrics.weight_hash,
            "parameter_count": eval_metrics.parameter_count,
            "checkpoint_file": "model.pt",
        }
        with open(ckpt_dir / "latest_checkpoint.json", "w", encoding="utf-8") as f:
            json.dump(latest_meta, f, indent=2)

        # 2. Copy tokenizer artifacts
        src_tok = Path(tokenizer_dir)
        for fname in ["vocab.json", "merges.json", "config.json"]:
            fpath = src_tok / fname
            if fpath.is_file():
                shutil.copy2(fpath, tok_dir / fname)

        # 3. Write configs
        with open(cfg_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(model_config_dict, f, indent=2)
        with open(cfg_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(training_config_dict, f, indent=2)

        # 4. Write evaluation summary
        with open(eval_dir / "eval_summary.json", "w", encoding="utf-8") as f:
            json.dump(eval_metrics.to_dict(), f, indent=2)

        # 5. Generate Model Card
        model_card_content = f"""# ChakrMicro Model Card: {release_tag}

## Model Overview
- **Model Identity**: ChakrMicro v0.1
- **Parameter Count**: {eval_metrics.parameter_count:,}
- **Architecture**: 6-layer Pre-LN Transformer, d_model=256, n_heads=8, d_ff=1024
- **Vocabulary Size**: {eval_metrics.tokenizer_vocab_size}
- **Context Length**: 512 tokens
- **Weight SHA-256**: `{eval_metrics.weight_hash}`
- **Canonical Baseline Status**: {"CANONICAL BASELINE" if eval_metrics.is_canonical_baseline else "TRAINED EXPERIMENTAL RELEASE"}

## Evaluation Metrics
- **Validation Loss**: {eval_metrics.val_loss:.4f}
- **Validation Perplexity**: {eval_metrics.val_perplexity:.4f}
- **Weights Finite**: {eval_metrics.weights_finite}
- **Inference Invariant (ΔW = 0)**: {eval_metrics.inference_delta_w_zero}

## Proven vs Unproven Capabilities
- Parameter Count Exactness: **{eval_metrics.proven_status.get('parameter_count_exact')}**
- Deterministic Inference: **{eval_metrics.proven_status.get('deterministic_inference')}**
- Inference ΔW = 0: **{eval_metrics.proven_status.get('inference_delta_w_zero')}**
- Open-Domain Fluency: **{eval_metrics.proven_status.get('open_domain_fluency')}**

## Honest Disclosures & Limitations
1. This is a 3.4M parameter micro-scale language model trained from scratch without external weights.
2. Open-domain conversational fluency and broad world-knowledge reasoning are explicitly **UNPROVEN**.
3. Do not deploy this model for open ungrounded decision making without cognitive grounding.
"""
        with open(target_dir / "MODEL_CARD.md", "w", encoding="utf-8") as f:
            f.write(model_card_content)

        # 6. Generate Checksums
        checksums = {}
        for p in sorted(target_dir.rglob("*")):
            if p.is_file() and p.name != "checksums.sha256":
                rel = p.relative_to(target_dir).as_posix()
                checksums[rel] = compute_file_sha256(p)

        with open(target_dir / "checksums.sha256", "w", encoding="utf-8") as f:
            for rel, h in checksums.items():
                f.write(f"{h}  {rel}\n")

        return target_dir
