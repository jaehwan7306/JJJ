"""Verified fallback for LLMLingua 0.2.2's tokenizer round-trip assertion.

Kept outside src/ so an existing run's source signature remains valid. The
adapter has its own version and source hash recorded/checked in run.json.
The original method is tried first: previously successful contexts are unchanged.
No text, token IDs, ranking settings, or compression budgets are modified.
"""
from __future__ import annotations

import hashlib
import sys
from functools import wraps
from pathlib import Path

FIX_ID = "llmlingua-prefix-offset-fallback-v1"


def prefix_boundary_from_offsets(tokenizer, prefix: str, text: str) -> int:
    """Locate a proven boundary in the actual combined token stream.

    Decoding is not used as an equality test: whitespace cleanup and tokenizer
    normalization can make decode(encode(text)) differ from the source string.
    A token crossing the prefix/context boundary is rejected, never guessed.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Prefix fallback requires a fast tokenizer with source offsets")
    combined = prefix + text
    encoded = tokenizer(combined, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if len(ids) != len(offsets):
        raise ValueError("Token IDs and source offsets disagree")
    if ids != tokenizer(combined, add_special_tokens=False)["input_ids"]:
        raise ValueError("Requesting source offsets changed tokenization")
    boundary = len(prefix)
    first_context = None
    for index, (start, end) in enumerate(offsets):
        if not 0 <= start <= end <= len(combined):
            raise ValueError("Tokenizer returned invalid source offsets")
        if start < boundary < end:
            raise ValueError("A token crosses the question/context boundary; no safe split exists")
        if end > boundary and first_context is None:
            first_context = index
        if first_context is not None and start < boundary:
            raise ValueError("Tokenizer offsets do not form a question/context partition")
    if first_context is None or first_context == 0:
        raise ValueError("Cannot verify both prefix and context tokens")
    return first_context


def apply_prefix_fix() -> dict:
    """Install an idempotent, checkpoint-audited compatibility adapter."""
    from llmlingua import PromptCompressor
    from contractnli.experiment import Experiment
    from contractnli.storage import write_json

    canonical_source = Path(__file__).read_text(encoding="utf-8")
    specification = {
        "id": FIX_ID,
        "implementation_sha256": hashlib.sha256(canonical_source.encode()).hexdigest(),
        "mode": "original method first; verified source offsets only on AssertionError",
    }
    installed = getattr(PromptCompressor.get_prefix_length, "_contractnli_prefix_fix", None)
    if installed is not None:
        if installed != specification:
            raise RuntimeError("Prefix compatibility code changed in this kernel; restart the kernel")
        return specification
    original_prefix_length = PromptCompressor.get_prefix_length
    original_open = Experiment._open

    @wraps(original_prefix_length)
    def get_prefix_length(self, prefix: str, text: str):
        try:
            return original_prefix_length(self, prefix, text)
        except AssertionError as original_error:
            try:
                return prefix_boundary_from_offsets(self.tokenizer, prefix, text)
            except ValueError as fallback_error:
                raise ValueError(f"Verified prefix fallback failed: {fallback_error}") from original_error

    @wraps(original_open)
    def open_with_compatibility_record(self):
        # Keep the original config/data/source signature and runtime checks intact.
        original_open(self)
        patches = self.manifest.setdefault("compatibility_patches", {})
        previous = patches.get(FIX_ID)
        if previous is not None:
            if previous.get("specification") != specification:
                raise ValueError("Saved prefix adapter hash differs; preserve this run and use the original adapter")
        else:
            patches[FIX_ID] = {
                "specification": specification,
                "contexts_before_registration": len(self.contexts.rows),
                "compressed_contexts_before_registration": sum(
                    row["condition"] == "longllmlingua" for row in self.contexts.rows.values()
                ),
                "predictions_before_registration": len(self.predictions.rows),
                "existing_successful_contexts_preserved": True,
            }
            write_json(self.manifest_path, self.manifest)

    get_prefix_length._contractnli_prefix_fix = specification
    PromptCompressor.get_prefix_length = get_prefix_length
    Experiment._open = open_with_compatibility_record
    return specification


def validate_prefix_case(config, document_id: str, hypothesis_id: str) -> dict:
    """Check every possible first ranked chunk, using cached tokenizer files only.

    No model weights are loaded and no ranking/inference is performed. This lets
    a user verify the exact failing case before resuming an expensive run.
    """
    from llmlingua import PromptCompressor
    from transformers import AutoConfig, AutoTokenizer
    from contractnli.compression import split_context
    from contractnli.data import load_dataset

    apply_prefix_fix()
    document = next((d for d in load_dataset(config.dataset).documents if d.id == str(document_id)), None)
    if document is None:
        raise ValueError(f"Document not found: {document_id}")
    example = next((e for e in document.examples if e.hypothesis_id == hypothesis_id), None)
    if example is None:
        raise ValueError(f"Hypothesis not found: {hypothesis_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.compressor.name, revision=config.compressor.revision,
                                               local_files_only=True, trust_remote_code=False)
    model_config = AutoConfig.from_pretrained(config.compressor.name, revision=config.compressor.revision,
                                              local_files_only=True, trust_remote_code=False)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    question = "Determine whether the following hypothesis is entailed, contradicted, or not mentioned: " + example.hypothesis
    window = min(config.compressor.max_input_tokens, model_config.max_position_embeddings)
    limit = min(512, window - count(question) - 402)
    chunks = split_context(document.text, count, limit)
    probe = PromptCompressor.__new__(PromptCompressor)
    probe.tokenizer = tokenizer
    for chunk in chunks:
        probe.get_prefix_length(question + "\n\n", chunk)
    return {"status": "PASS", "document_id": str(document_id), "hypothesis_id": hypothesis_id,
            "checked_candidate_chunks": len(chunks), "model_weights_loaded": False}


if __name__ == "__main__":
    from contractnli.cli import main

    print("Applying", apply_prefix_fix()["id"], flush=True)
    main(["run", *sys.argv[1:]])
