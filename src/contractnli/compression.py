from __future__ import annotations

import time
from dataclasses import dataclass

from .backend import TransformersBackend
from .config import ModelConfig


def split_context(text: str, count_tokens, limit: int = 512) -> list[str]:
    """Deterministic character-preserving chunks; no gold evidence or hypotheses used."""
    if limit < 2:
        raise ValueError("Chunk budget too small")
    chunks: list[str] = []
    pending = ""
    for line in text.splitlines(keepends=True):
        if pending and count_tokens(pending + line) > limit:
            chunks.append(pending)
            pending = ""
        while count_tokens(line) > limit:
            low, high = 1, len(line)
            while low < high:
                middle = (low + high + 1) // 2
                if count_tokens(line[:middle]) <= limit:
                    low = middle
                else:
                    high = middle - 1
            if count_tokens(line[:low]) > limit:
                raise ValueError("Cannot split within token budget")
            chunks.append(line[:low])
            line = line[low:]
        pending += line
    if pending:
        chunks.append(pending)
    return chunks


class LegacyCacheAdapter:
    """Bridge LLMLingua 0.2.2 list caches and Transformers' DynamicCache.

    Only cache representation changes; tensors and model logits are unchanged.
    Kept separate so the compatibility path can be tested with a tiny random Llama.
    """

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        from transformers.cache_utils import DynamicCache

        past = kwargs.get("past_key_values")
        if past is not None and isinstance(past, (tuple, list)):
            kwargs["past_key_values"] = DynamicCache.from_legacy_cache(tuple(tuple(v) for v in past))
        result = self.model(*args, **kwargs)
        if hasattr(result.past_key_values, "to_legacy_cache"):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result


@dataclass(frozen=True)
class Compression:
    text: str
    seconds: float
    scorer_tokens: int
    target_scorer_tokens: int


class LongLLMLinguaBackend:
    def __init__(self, config: ModelConfig, seed: int):
        from llmlingua import PromptCompressor

        self.backend = TransformersBackend(config, seed)
        self.config = config
        backend = self.backend

        class ReuseModelCompressor(PromptCompressor):
            def load_model(self, model_name, device_map="cuda", model_config=None):
                self.model = LegacyCacheAdapter(backend.model)
                self.tokenizer = backend.tokenizer
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.device = str(backend.model.device)
                self.context_idxs = []
                self.max_position_embeddings = min(config.max_input_tokens,
                                                     backend.model.config.max_position_embeddings)

        self.compressor = ReuseModelCompressor(model_name=config.name, device_map=config.device)
        self.identity = {**self.backend.identity, "algorithm": "longllmlingua",
                         "chunk_tokens": 512, "context_budget": "+100",
                         "dynamic_context_compression_ratio": 0.3,
                         "cache_adapter": "legacy-to-dynamic-v1"}

    def compress(self, text: str, hypothesis: str, retention_ratio: float) -> Compression:
        if self.config.device == "cuda":
            self.backend.torch.cuda.synchronize()
        start = time.perf_counter()
        question = "Determine whether the following hypothesis is entailed, contradicted, or not mentioned: " + hypothesis
        question_tokens = self.backend.count_tokens(question)
        chunk_limit = min(512, self.compressor.max_position_embeddings - question_tokens - 402)
        if chunk_limit < 2:
            raise ValueError("Question too long for LongLLMLingua scoring window; no truncation performed")
        chunks = split_context(text, self.backend.count_tokens, chunk_limit)
        target = max(1, int(self.backend.count_tokens(text) * retention_ratio))
        result = self.compressor.compress_prompt(
            chunks, question=question, target_token=target,
            condition_in_question="after_condition", reorder_context="sort",
            dynamic_context_compression_ratio=0.3, condition_compare=True,
            context_budget="+100", rank_method="longllmlingua",
            use_context_level_filter=True, use_token_level_filter=True,
            use_sentence_level_filter=False, concate_question=False,
        )
        # concate_question=False is essential: the judge receives the hypothesis separately.
        compressed = result["compressed_prompt"].strip()
        if not compressed:
            raise ValueError("LongLLMLingua returned empty context")
        scorer_tokens = self.backend.count_tokens(compressed)
        if self.config.device == "cuda":
            self.backend.torch.cuda.synchronize()
        return Compression(compressed, time.perf_counter() - start, scorer_tokens, target)

    def close(self):
        self.compressor = None
        self.backend.close()
