from __future__ import annotations

import tomllib
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
    output_dir: str = "outputs/base"
    conditions: tuple[str, ...] = ("original", "naive", "contextualized", "longllmlingua")
    retention_ratio: float = 0.30
    judge_max_new_tokens: int = 16
    seed: int = 42
    max_documents: int | None = None
    summarizer: ModelConfig = field(default_factory=lambda: ModelConfig("Qwen/Qwen2.5-7B-Instruct"))
    judge: ModelConfig = field(default_factory=lambda: ModelConfig("Qwen/Qwen3-8B"))
    compressor: ModelConfig = field(default_factory=lambda: ModelConfig("NousResearch/Llama-2-7b-hf", max_input_tokens=4096))

    def __post_init__(self):
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions must be non-empty and unique")
        if set(self.conditions) - {"original", "naive", "contextualized", "longllmlingua"}:
            raise ValueError("Unknown experimental condition")
        if not 0 < self.retention_ratio < 1 or self.judge_max_new_tokens < 1:
            raise ValueError("retention_ratio must be in (0,1); generation budget must be positive")
        if self.max_documents is not None and self.max_documents < 1:
            raise ValueError("max_documents must be positive")

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    for name in ("summarizer", "judge", "compressor"):
        if name in raw:
            raw[name] = ModelConfig(**raw[name])
    if "conditions" in raw:
        raw["conditions"] = tuple(raw["conditions"])
    return Config(**raw)
