"""Tokenizer-only input-size checks before allocating model weights."""
from __future__ import annotations

from .compression import split_context
from .data import load_dataset
from .prompts import judge_messages, summary_messages


def preflight(config):
    from transformers import AutoTokenizer

    dataset = load_dataset(config.dataset, config.max_documents)
    tokenizers = {}
    for role in ("judge", "summarizer", "compressor"):
        if role == "summarizer" and not set(config.conditions) & {"naive", "contextualized"}:
            continue
        if role == "compressor" and "longllmlingua" not in config.conditions:
            continue
        model = getattr(config, role)
        tokenizers[role] = AutoTokenizer.from_pretrained(model.name, revision=model.revision,
                                                        trust_remote_code=False)
    reference = tokenizers["judge"]
    problems = []
    maxima = {}
    for doc in dataset.documents:
        original_tokens = len(reference.encode(doc.text, add_special_tokens=False))
        target = max(1, int(original_tokens * config.retention_ratio))
        if "summarizer" in tokenizers:
            for method in set(config.conditions) & {"naive", "contextualized"}:
                tokens = tokenizers["summarizer"].apply_chat_template(
                    summary_messages(doc.text, method, target), tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
                required = len(tokens) + target
                maxima["summary_prompt_plus_output_tokens"] = max(required, maxima.get("summary_prompt_plus_output_tokens", 0))
                if required > config.summarizer.max_input_tokens:
                    problems.append({"document_id": doc.id, "role": "summarizer", "tokens": required})
        if "original" in config.conditions:
            for example in doc.examples:
                tokens = reference.apply_chat_template(judge_messages(doc.text, example.hypothesis),
                                                       tokenize=True, add_generation_prompt=True,
                                                       enable_thinking=False)
                required = len(tokens) + config.judge_max_new_tokens
                maxima["original_judge_prompt_plus_output_tokens"] = max(required, maxima.get("original_judge_prompt_plus_output_tokens", 0))
                if required > config.judge.max_input_tokens:
                    problems.append({"example_id": example.key, "role": "judge", "tokens": required})
        if "compressor" in tokenizers:
            count = lambda text: len(tokenizers["compressor"].encode(text, add_special_tokens=False))
            chunks = split_context(doc.text, count, 512)
            maxima["longllmlingua_chunks_per_document"] = max(len(chunks), maxima.get("longllmlingua_chunks_per_document", 0))
    return {"ok": not problems, "maxima": maxima, "context_overflows": problems,
            "note": "Tokenizer-only check. GPU availability, 4bit loading and peak VRAM require an actual smoke run."}
