from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from . import prompts
from .backend import TransformersBackend
from .compression import LongLLMLinguaBackend
from .config import Config
from .data import Dataset, LABELS, load_dataset
from .metrics import classification, mean, paired_metrics
from .storage import JsonlStore, digest, write_json


def inspect(config: Config, dataset: Dataset) -> dict:
    return {
        "documents": len(dataset.documents), "evaluated_pairs": len(dataset.examples),
        "gold_label_counts": dict(Counter(e.gold for e in dataset.examples)),
        "gold_evidence_available_pairs": sum(bool(e.evidence) for e in dataset.examples),
        "not_mentioned_pairs": dataset.not_mentioned_pairs,
        "conditions": list(config.conditions), "target_token_retention": config.retention_ratio,
        "summary_requests": len(dataset.documents) * len(set(config.conditions) & {"naive", "contextualized"}),
        "question_aware_compression_requests": len(dataset.examples) if "longllmlingua" in config.conditions else 0,
        "judgment_requests": len(dataset.examples) * len(config.conditions),
        "dataset_sha256": dataset.sha256,
        "partial_dataset": config.max_documents is not None,
    }


def versions() -> dict:
    result = {"python": platform.python_version()}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes", "llmlingua", "tokenizers"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


@contextmanager
def exclusive_run(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".run.lock"
    try:
        with lock.open("x") as handle:
            handle.write(f"{platform.node()} pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise RuntimeError(f"Another process or interrupted run owns {lock}. Verify it stopped before removing the lock.") from exc
    try:
        yield
    finally:
        lock.unlink()


class ReferenceTokenizer:
    def __init__(self, config):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(config.name, revision=config.revision,
                                                       trust_remote_code=False)
        self.identity = {"name": config.name, "revision": config.revision,
                         "resolved_revision": self.tokenizer.init_kwargs.get("_commit_hash")}

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class Experiment:
    def __init__(self, config: Config, backend_factory=TransformersBackend,
                 compressor_factory=LongLLMLinguaBackend, tokenizer_factory=ReferenceTokenizer):
        self.config = config
        self.dataset = load_dataset(config.dataset, config.max_documents)
        if not self.dataset.examples:
            raise ValueError("No evaluation examples")
        self.directory = Path(config.output_dir)
        self.backend_factory = backend_factory
        self.compressor_factory = compressor_factory
        self.tokenizer_factory = tokenizer_factory

    def _open(self):
        config = self.config.to_dict()
        # Paths may change when copying checkpoints between local PCs.
        config.pop("dataset")
        config.pop("output_dir")
        code = {p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.glob("*.py")}
        signature = digest({"config": config, "dataset": self.dataset.sha256, "source": code})
        self.manifest_path = self.directory / "run.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest["signature"] != signature:
                raise ValueError("Config/data/source changed. Use a new output_dir; refusing stale checkpoint reuse.")
        else:
            if any(self.directory.glob("*.json*")):
                raise ValueError("Output files exist without run.json; use a fresh output directory")
            self.manifest = {"signature": signature, "config": self.config.to_dict(),
                             "dataset": inspect(self.config, self.dataset), "stages": {}, "models": {}}
            write_json(self.manifest_path, self.manifest)
        self.contexts = JsonlStore(self.directory / "contexts.jsonl")
        self.predictions = JsonlStore(self.directory / "predictions.jsonl")

    def _identity(self, role, identity):
        # Persist resolved revisions, precision, hardware and package versions per role.
        recorded = {**identity, "packages": versions()}
        previous = self.manifest["models"].get(role)
        if previous is not None and previous != recorded:
            raise ValueError(f"{role} model/runtime changed. Use original runtime or a new output_dir.")
        self.manifest["models"][role] = recorded
        write_json(self.manifest_path, self.manifest)

    def _context_key(self, method, document, example=None):
        suffix = f"/{example.hypothesis_id}" if method == "longllmlingua" else ""
        return f"{method}/{document.id}{suffix}"

    def _ensure_tokenizer(self):
        if not hasattr(self, "tokenizer"):
            self.tokenizer = self.tokenizer_factory(self.config.judge)
            self._identity("reference_tokenizer", self.tokenizer.identity)

    def _save_context(self, key, method, doc, text, seconds, **extra):
        original_tokens = self.tokenizer.count_tokens(doc.text)
        tokens = self.tokenizer.count_tokens(text)
        if not tokens:
            raise ValueError(f"Empty generated context: {key}")
        self.contexts.add(key, {"condition": method, "document_id": doc.id, "text": text,
                              "original_tokens": original_tokens, "context_tokens": tokens,
                              "target_reference_tokens": max(1, int(original_tokens * self.config.retention_ratio)),
                              "actual_token_retention": tokens / original_tokens,
                              "compression_seconds": seconds, **extra})

    def summarize(self):
        methods = [m for m in self.config.conditions if m in ("naive", "contextualized")]
        pending = [(m, d) for d in self.dataset.documents for m in methods
                   if self._context_key(m, d) not in self.contexts.rows]
        if not pending:
            return
        self._ensure_tokenizer()
        backend = self.backend_factory(self.config.summarizer, self.config.seed)
        try:
            self._identity("summarizer", backend.identity)
            for i, (method, doc) in enumerate(pending, 1):
                target = max(1, int(self.tokenizer.count_tokens(doc.text) * self.config.retention_ratio))
                generated = backend.generate(prompts.summary_messages(doc.text, method, target), target)
                self._save_context(self._context_key(method, doc), method, doc, generated.text, generated.seconds,
                                   generation=asdict(generated))
                print(f"summary {i}/{len(pending)}: {method}/{doc.id}", flush=True)
        finally:
            backend.close()

    def compress(self):
        if "longllmlingua" not in self.config.conditions:
            return
        pending = [(d, e) for d in self.dataset.documents for e in d.examples
                   if self._context_key("longllmlingua", d, e) not in self.contexts.rows]
        if not pending:
            return
        self._ensure_tokenizer()
        backend = self.compressor_factory(self.config.compressor, self.config.seed)
        try:
            self._identity("compressor", backend.identity)
            for i, (doc, example) in enumerate(pending, 1):
                compressed = backend.compress(doc.text, example.hypothesis, self.config.retention_ratio)
                self._save_context(self._context_key("longllmlingua", doc, example), "longllmlingua", doc,
                                   compressed.text, compressed.seconds, hypothesis_id=example.hypothesis_id,
                                   scorer_tokens=compressed.scorer_tokens,
                                   target_scorer_tokens=compressed.target_scorer_tokens)
                print(f"compression {i}/{len(pending)}: {example.key}", flush=True)
        finally:
            backend.close()

    def judge(self):
        # Validate prerequisites BEFORE loading the judge or writing partial original results.
        work = []
        for method in self.config.conditions:
            for doc in self.dataset.documents:
                for example in doc.examples:
                    key = f"{method}/{example.key}"
                    if key in self.predictions.rows:
                        continue
                    context_key = self._context_key(method, doc, example)
                    if method != "original" and context_key not in self.contexts.rows:
                        raise ValueError(f"Missing {context_key}: run summarize/compress stages first")
                    work.append((method, doc, example, key, context_key))
        if not work:
            return
        self._ensure_tokenizer()
        backend = self.backend_factory(self.config.judge, self.config.seed)
        try:
            self._identity("judge", backend.identity)
            for i, (method, doc, example, key, context_key) in enumerate(work, 1):
                context = doc.text if method == "original" else self.contexts.rows[context_key]["text"]
                generated = backend.generate(prompts.judge_messages(context, example.hypothesis),
                                             self.config.judge_max_new_tokens)
                self.predictions.add(key, {
                    "condition": method, "example_id": example.key, "document_id": doc.id,
                    "hypothesis_id": example.hypothesis_id, "hypothesis": example.hypothesis,
                    "gold": example.gold, "prediction": prompts.parse_label(generated.text),
                    "gold_evidence": list(example.evidence), "has_gold_evidence": bool(example.evidence),
                    "context_key": context_key if method != "original" else None,
                    "context_tokens": self.tokenizer.count_tokens(context),
                    "original_tokens": self.tokenizer.count_tokens(doc.text),
                    "generation": asdict(generated),
                })
                print(f"judgment {i}/{len(work)}: {key}", flush=True)
        finally:
            backend.close()

    def report(self):
        expected = {f"{m}/{e.key}" for m in self.config.conditions for e in self.dataset.examples}
        if self.predictions.rows.keys() != expected:
            missing = expected - self.predictions.rows.keys()
            raise ValueError(f"Cannot report incomplete/foreign predictions; missing={len(missing)}")
        rows = list(self.predictions.rows.values())
        grouped = {m: [r for r in rows if r["condition"] == m] for m in self.config.conditions}
        metrics = {}
        for method, predictions in grouped.items():
            contexts = [c for c in self.contexts.rows.values() if c["condition"] == method]
            compression_seconds = sum(c["compression_seconds"] for c in contexts)
            judgment_seconds = sum(r["generation"]["seconds"] for r in predictions)
            total_seconds = compression_seconds + judgment_seconds
            result = classification(predictions)
            result.update({
                "compression_calls": len(contexts), "compression_seconds": compression_seconds,
                "judgment_seconds": judgment_seconds, "total_inference_seconds": total_seconds,
                "amortized_seconds_per_pair": total_seconds / len(predictions),
                "mean_actual_token_retention": mean(r["context_tokens"] / r["original_tokens"] for r in predictions),
                "corpus_token_retention": sum(r["context_tokens"] for r in predictions) / sum(r["original_tokens"] for r in predictions),
                "over_target_context_calls": sum(c["context_tokens"] > c["target_reference_tokens"] for c in contexts),
                "length_capped_summary_calls": sum(c.get("generation", {}).get("finish_reason") == "length" for c in contexts),
            })
            if method != "original" and "original" in grouped:
                result.update(paired_metrics(grouped["original"], predictions))
            metrics[method] = result
        report = {"complete": True, "signature": self.manifest["signature"],
                  "dataset": inspect(self.config, self.dataset), "metrics": metrics,
                  "notes": ["Three gold classes retained, including NotMentioned.",
                            "No semantic gold-evidence preservation score is computed in the base experiment.",
                            "Token counts use the judge tokenizer; configured retention is a target, not guaranteed output length.",
                            "Inference time includes per-call tokenization and compression/generation, excludes model loading and checkpoint I/O.",
                            "Repeated runs reuse stored timings; they are not new latency measurements."]}
        write_json(self.directory / "metrics.json", report)
        with (self.directory / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["condition", "n", "accuracy", "macro_f1", "decision_flip_rate", "decision_agreement_rate",
                      "correct_decision_retention", "mean_actual_token_retention", "total_inference_seconds",
                      "invalid_outputs"]
            writer = csv.DictWriter(handle, fields, extrasaction="ignore")
            writer.writeheader()
            for method, result in metrics.items():
                writer.writerow({"condition": method, **result})
        return report

    def run(self, stage="all"):
        if stage not in ("all", "summarize", "compress", "judge", "report"):
            raise ValueError(f"Unknown stage: {stage}")
        with exclusive_run(self.directory):
            self._open()
            started = time.perf_counter()
            result = None
            for name in ("summarize", "compress", "judge", "report"):
                if stage in ("all", name):
                    result = getattr(self, name)()
            self.manifest["stages"][stage] = {"last_invocation_wall_seconds": time.perf_counter() - started,
                                              "completed": True}
            write_json(self.manifest_path, self.manifest)
            return result
