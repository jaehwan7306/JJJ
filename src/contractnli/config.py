from __future__ import annotations

import tomllib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    name: str
    revision: str = "main"
    device: str = "cuda"
    dtype: str = "bfloat16"
    quantization: str = "4bit"
    max_input_tokens: int = 32768

    def __post_init__(self):
        if not self.name or not self.revision:
            raise ValueError("Model name and revision cannot be empty")
        if self.device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError("Unsupported dtype")
        if self.quantization not in ("none", "4bit"):
            raise ValueError("quantization must be none or 4bit")
        if self.quantization == "4bit" and self.device != "cuda":
            raise ValueError("4bit quantization requires CUDA")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("Use float32 for CPU inference")
        if self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be positive")


@dataclass(frozen=True)
class Config:
    dataset: str = "data/contract-nli/test.json"
    expected_dataset_sha256: str | None = None
    output_dir: str = "outputs/base"
    conditions: tuple[str, ...] = ("original", "naive", "contextualized", "longllmlingua")
    retention_ratio: float = 0.30
    judge_max_new_tokens: int = 16
    seed: int = 42
    max_documents: int | None = None
    judge_decoding: str = "free"
    budget_policy: str = "target"
    budget_attempt_limit: int = 6
    warmup_calls: int = 0
    shuffle_judgments: bool = False
    require_pinned_revisions: bool = False
    require_freeze: bool = False
    summary_length_policy: str = "ratio"
    summary_max_new_tokens: int = 2048
    compressor_budget_mode: str = "target_tokens"
    summarizer: ModelConfig = field(default_factory=lambda: ModelConfig("Qwen/Qwen2.5-7B-Instruct"))
    judge: ModelConfig = field(default_factory=lambda: ModelConfig("Qwen/Qwen3-8B"))
    compressor: ModelConfig = field(default_factory=lambda: ModelConfig("NousResearch/Llama-2-7b-hf", max_input_tokens=4096))

    def __post_init__(self):
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions must be non-empty and unique")
        if self.expected_dataset_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.expected_dataset_sha256):
            raise ValueError("expected_dataset_sha256 must be an immutable SHA256")
        if set(self.conditions) - {"original", "naive", "contextualized", "longllmlingua"}:
            raise ValueError("Unknown experimental condition")
        if not 0 < self.retention_ratio < 1 or self.judge_max_new_tokens < 1:
            raise ValueError("retention_ratio must be in (0,1); generation budget must be positive")
        if self.max_documents is not None and self.max_documents < 1:
            raise ValueError("max_documents must be positive")
        if self.judge_decoding not in ("free", "constrained"):
            raise ValueError("judge_decoding must be free or constrained")
        if self.budget_policy not in ("target", "reference_cap"):
            raise ValueError("budget_policy must be target or reference_cap")
        if self.budget_attempt_limit < 1 or self.warmup_calls < 0:
            raise ValueError("budget_attempt_limit must be positive; warmup_calls cannot be negative")
        if self.summary_length_policy not in ("ratio", "fixed") or self.summary_max_new_tokens < 1:
            raise ValueError("summary_length_policy must be ratio/fixed with a positive summary_max_new_tokens")
        if self.compressor_budget_mode not in ("target_tokens", "rate"):
            raise ValueError("compressor_budget_mode must be target_tokens or rate")
        if self.budget_policy == "reference_cap" and self.summary_length_policy != "ratio":
            raise ValueError("reference_cap is a separate length-controlled protocol; use ratio summaries")
        if self.require_pinned_revisions:
            roles = [self.judge]
            if set(self.conditions) & {"naive", "contextualized"}:
                roles.append(self.summarizer)
            if "longllmlingua" in self.conditions:
                roles.append(self.compressor)
            if any(not re.fullmatch(r"[0-9a-f]{40}", m.revision) for m in roles):
                raise ValueError("Freeze model revisions to immutable 40-character HF commits before running")

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as handle:
        raw = json.load(handle) if Path(path).suffix == ".json" else tomllib.load(handle)
    for name in ("summarizer", "judge", "compressor"):
        if name in raw:
            raw[name] = ModelConfig(**raw[name])
    if "conditions" in raw:
        raw["conditions"] = tuple(raw["conditions"])
    return Config(**raw)
