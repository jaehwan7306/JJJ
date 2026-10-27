from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Protocol

from .config import ModelConfig
from .data import LABELS


class LabelConstraint:
    """Token trie for complete labels followed by EOS, never arbitrary prose."""

    def __init__(self, tokenizer, prompt_length, eos_token_id, max_new_tokens):
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id
        self.paths = [tuple(tokenizer.encode(label, add_special_tokens=False)) for label in LABELS]
        if eos_token_id is None or any(not path or eos_token_id in path for path in self.paths):
            raise ValueError("Labels require nonempty, non-EOS token sequences")
        if len(set(self.paths)) != len(LABELS):
            raise ValueError("Label token sequences are not distinct")
        if any(tokenizer.decode(list(path), skip_special_tokens=True).strip() != label
               for path, label in zip(self.paths, LABELS)):
            raise ValueError("Label tokenization does not round-trip")
        if max(map(len, self.paths)) + 1 > max_new_tokens:
            raise ValueError("judge_max_new_tokens must fit every complete label plus EOS")

    def __call__(self, batch_id, input_ids):
        suffix = tuple(input_ids[self.prompt_length:].tolist())
        allowed = {path[len(suffix)] for path in self.paths
                   if len(path) > len(suffix) and path[:len(suffix)] == suffix}
        if suffix in self.paths:
            allowed.add(self.eos_token_id)
        if not allowed:
            raise RuntimeError("Generated sequence left the allowed label trie")
        return sorted(allowed)


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    finish_reason: str


class Backend(Protocol):
    identity: dict

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> Generation: ...
    def generate_labels(self, messages: list[dict[str, str]], max_new_tokens: int) -> Generation: ...
    def count_tokens(self, text: str) -> int: ...
    def close(self) -> None: ...


class TransformersBackend:
    """One model at a time, greedy decoding, no silent input truncation."""

    def __init__(self, config: ModelConfig, seed: int):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        except ImportError as exc:
            raise RuntimeError('Install inference dependencies: pip install -e ".[inference]"') from exc
        self.torch = torch
        self.config = config
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not found. Run on a GPU host, or explicitly configure CPU/float32.")
        if config.device == "cuda" and config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support bfloat16; configure float16.")
        set_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.tokenizer = AutoTokenizer.from_pretrained(config.name, revision=config.revision,
                                                       trust_remote_code=False)
        kwargs = {"revision": config.revision, "trust_remote_code": False,
                  "torch_dtype": getattr(torch, config.dtype),
                  "attn_implementation": "sdpa", "low_cpu_mem_usage": True,
                  "device_map": {"": 0 if config.device == "cuda" else "cpu"}}
        if config.quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, config.dtype))
        try:
            self.model = AutoModelForCausalLM.from_pretrained(config.name, **kwargs).eval()
        except BaseException:
            self.close()
            raise
        tokenizer_commit = self.tokenizer.init_kwargs.get("_commit_hash")
        model_commit = getattr(self.model.config, "_commit_hash", None)
        if tokenizer_commit and model_commit and tokenizer_commit != model_commit:
            self.close()
            raise RuntimeError("Tokenizer/model revisions differ. Pin the same immutable HF commit.")
        self.identity = {"name": config.name, "requested_revision": config.revision,
                         "resolved_revision": model_commit,
                         "tokenizer_revision": tokenizer_commit,
                         "device": str(self.model.device), "dtype": config.dtype,
                         "quantization": config.quantization,
                         "attention_implementation": "sdpa", "inference_batch_size": 1,
                         "device_name": torch.cuda.get_device_name(0) if config.device == "cuda" else "cpu",
                         "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                         "tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
                         "gpu_capability": list(torch.cuda.get_device_capability(0)) if config.device == "cuda" else None,
                         "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if config.device == "cuda" else None}

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> Generation:
        return self._generate(messages, max_new_tokens, constrained=False)

    def generate_labels(self, messages: list[dict[str, str]], max_new_tokens: int) -> Generation:
        return self._generate(messages, max_new_tokens, constrained=True)

    def _generate(self, messages, max_new_tokens, constrained):
        from transformers import GenerationConfig

        if self.config.device == "cuda":
            self.torch.cuda.synchronize()
        wall_start = time.perf_counter()
        # Qwen3: explicitly disable thinking. For Qwen2.5 this extra template kwarg is unused.
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt", return_token_type_ids=False)
        prompt_tokens = inputs["input_ids"].shape[1]
        architecture_limit = getattr(self.model.config, "max_position_embeddings", None)
        context_limit = min(self.config.max_input_tokens, architecture_limit or self.config.max_input_tokens)
        if prompt_tokens + max_new_tokens > context_limit:
            raise ValueError(f"Context overflow: {prompt_tokens} prompt + {max_new_tokens} output > "
                             f"{context_limit}. No truncation was performed; use a supported larger context.")
        inputs = inputs.to(self.model.device)
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        kwargs = {}
        if constrained:
            eos = self.tokenizer.eos_token_id
            eos_ids = {eos}
            kwargs["prefix_allowed_tokens_fn"] = LabelConstraint(self.tokenizer, prompt_tokens, eos, max_new_tokens)
        # Build a clean greedy config; do not inherit sampling/forced-token settings.
        generation_config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                              eos_token_id=eos, pad_token_id=self.tokenizer.eos_token_id,
                                              bos_token_id=self.tokenizer.bos_token_id, use_cache=True)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, generation_config=generation_config, **kwargs)
        if self.config.device == "cuda":
            self.torch.cuda.synchronize()
        tokens = outputs[0, prompt_tokens:]
        ids = tokens.tolist()
        stopped = bool(ids and ids[-1] in eos_ids)
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if constrained and (text not in LABELS or not stopped):
            raise RuntimeError(f"Constrained judge failed to produce a complete allowed label: {text!r}")
        return Generation(text, prompt_tokens, len(ids), time.perf_counter() - wall_start,
                          "eos" if stopped else "length")

    def close(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.config.device == "cuda":
            self.torch.cuda.empty_cache()
