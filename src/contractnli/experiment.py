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
from . import SOURCE_AT_IMPORT
from .backend import TransformersBackend
from .budget import fit_reference_cap
from .compression import LongLLMLinguaBackend
from .config import Config
from .data import Dataset, load_for_config
from .metrics import classification, mean, paired_metrics, quantile
from .storage import JsonlStore, digest, write_json


def inspect(config: Config, dataset: Dataset) -> dict:
    return {
        "documents": len(dataset.documents), "evaluated_pairs": len(dataset.examples),
        "gold_label_counts": dict(Counter(e.gold for e in dataset.examples)),
        "gold_evidence_available_pairs": sum(bool(e.evidence) for e in dataset.examples),
        "not_mentioned_pairs": dataset.not_mentioned_pairs,
        "conditions": list(config.conditions),
        "target_token_retention": config.retention_ratio if config.summary_length_policy == "ratio" else None,
        "summary_length_policy": config.summary_length_policy,
        "summary_max_new_tokens": config.summary_max_new_tokens if config.summary_length_policy == "fixed" else None,
        "compressor_budget_mode": config.compressor_budget_mode,
        "compressor_rate": config.retention_ratio if config.compressor_budget_mode == "rate" else None,
        "compressor_context_target_ratio": config.retention_ratio if config.compressor_budget_mode == "target_tokens" else None,
        "judge_decoding": config.judge_decoding, "budget_policy": config.budget_policy,
        "summary_requests": len(dataset.documents) * len(set(config.conditions) & {"naive", "contextualized"}),
        "question_aware_compression_requests": len(dataset.examples) if "longllmlingua" in config.conditions else 0,
        "judgment_requests": len(dataset.examples) * len(config.conditions),
        "dataset_sha256": dataset.sha256,
        "partial_dataset": config.max_documents is not None,
    }


def versions() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
              "execution_host": platform.node()}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes", "llmlingua", "tokenizers",
                    "huggingface_hub", "safetensors", "sentencepiece", "protobuf", "numpy"):
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
        if config.require_freeze and not config.require_pinned_revisions:
            raise ValueError("Freeze the template first; run the immutable frozen JSON config, not the TOML template")
        self.dataset = load_for_config(config)
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
        if code != SOURCE_AT_IMPORT:
            raise ValueError("Source files changed in this Python session. Restart the Jupyter kernel before running.")
        signature = digest({"config": config, "dataset": self.dataset.sha256, "source": code})
        self.manifest_path = self.directory / "run.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest["signature"] != signature:
                raise ValueError("Config/data/source changed. Use a new output_dir; refusing stale checkpoint reuse.")
        else:
            if any(self.directory.glob("*.json*")):
                raise ValueError("Output files exist without run.json; use a fresh output directory")
            self.manifest = {"schema_version": 2, "signature": signature, "config": self.config.to_dict(),
                             "dataset": inspect(self.config, self.dataset), "stages": {}, "models": {}}
            write_json(self.manifest_path, self.manifest)
        self.contexts = JsonlStore(self.directory / "contexts.jsonl")
        self.predictions = JsonlStore(self.directory / "predictions.jsonl")
        self._validate_checkpoints()

    def _validate_checkpoints(self):
        docs = {d.id: d for d in self.dataset.documents}
        examples = {e.key: e for e in self.dataset.examples}
        for key, row in self.contexts.rows.items():
            method, doc_id = row.get("condition"), row.get("document_id")
            if method not in self.config.conditions or method == "original" or doc_id not in docs:
                raise ValueError(f"Foreign context checkpoint: {key}")
            example = None
            if method == "longllmlingua":
                example = examples.get(f"{doc_id}/{row.get('hypothesis_id')}")
                if example is None:
                    raise ValueError(f"Foreign context hypothesis: {key}")
            if (key != self._context_key(method, docs[doc_id], example) or not row.get("text", "").strip()
                    or row.get("text_fingerprint") != digest({"text": row["text"]})):
                raise ValueError(f"Invalid context checkpoint: {key}")
        for key, row in self.predictions.rows.items():
            method, example_id = row.get("condition"), row.get("example_id")
            example = examples.get(example_id)
            if method not in self.config.conditions or example is None or key != f"{method}/{example_id}":
                raise ValueError(f"Foreign prediction checkpoint: {key}")
            if (row.get("gold"), row.get("document_id"), row.get("hypothesis_id"), row.get("hypothesis")) != (
                    example.gold, example.document_id, example.hypothesis_id, example.hypothesis):
                raise ValueError(f"Prediction metadata differs from dataset: {key}")
            parsed = prompts.parse_label(row["generation"]["text"])
            if parsed != row.get("prediction") or (self.config.judge_decoding == "constrained" and parsed is None):
                raise ValueError(f"Prediction output/parser mismatch: {key}")
            expected_context = None if method == "original" else self._context_key(method, docs[example.document_id], example)
            if row.get("context_key") != expected_context or (expected_context and expected_context not in self.contexts.rows):
                raise ValueError(f"Prediction context missing or mismatched: {key}")
            text = docs[example.document_id].text if expected_context is None else self.contexts.rows[expected_context]["text"]
            if row.get("context_fingerprint") != digest({"text": text}):
                raise ValueError(f"Prediction context text changed: {key}")
            if expected_context:
                context = self.contexts.rows[expected_context]
                if any(row[k] != context[k] for k in ("original_tokens", "context_tokens")):
                    raise ValueError(f"Prediction/context token counts disagree: {key}")
            elif row["original_tokens"] != row["context_tokens"]:
                raise ValueError(f"Original context was shortened: {key}")

    def _identity(self, role, identity):
        # Persist resolved revisions, precision, hardware and package versions per role.
        recorded = {**identity, "packages": versions()}
        if "device_name" in identity:
            environment = {k: recorded.get(k) for k in ("device_name", "gpu_capability", "gpu_total_bytes", "torch_cuda", "cudnn", "packages")}
            if self.manifest.get("inference_environment", environment) != environment:
                raise ValueError("Hardware/runtime changed between model roles. Use the same PC/runtime or a new output_dir.")
            self.manifest["inference_environment"] = environment
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
                              "text_fingerprint": digest({"text": text}),
                              "original_tokens": original_tokens, "context_tokens": tokens,
                              "target_reference_tokens": (max(1, int(original_tokens * self.config.retention_ratio))
                                                          if self.config.summary_length_policy == "ratio" else None),
                              "actual_token_retention": tokens / original_tokens,
                              "compression_seconds": seconds, **extra})

    def _warmup(self, role, call):
        started = time.perf_counter()
        for _ in range(self.config.warmup_calls):
            call()
        if self.config.warmup_calls:
            self.manifest.setdefault("warmups", []).append({"role": role, "calls": self.config.warmup_calls,
                                                           "wall_seconds": time.perf_counter() - started,
                                                           "excluded_from_inference_metrics": True})
            write_json(self.manifest_path, self.manifest)

    def _load_backend(self, role, factory, config):
        print(f"Loading {role}: {config.name} (download/loading time is recorded separately)", flush=True)
        started = time.perf_counter()
        backend = factory(config, self.config.seed)
        try:
            self.manifest.setdefault("model_preparation", []).append({"role": role,
                "wall_seconds": time.perf_counter() - started, "may_include_download": True,
                "excluded_from_inference_metrics": True})
            write_json(self.manifest_path, self.manifest)
        except BaseException:
            backend.close()
            raise
        return backend

    def summarize(self):
        methods = [m for m in self.config.conditions if m in ("naive", "contextualized")]
        pending = [(m, d) for d in self.dataset.documents for m in methods
                   if self._context_key(m, d) not in self.contexts.rows]
        if not pending:
            return
        self._ensure_tokenizer()
        backend = self._load_backend("summarizer", self.backend_factory, self.config.summarizer)
        try:
            self._identity("summarizer", backend.identity)
            self._warmup("summarizer", lambda: backend.generate(prompts.summary_messages(
                "This sample agreement protects confidential records and permits legally required disclosure.",
                "naive", 32), 32))
            for i, (method, doc) in enumerate(pending, 1):
                target = (max(1, int(self.tokenizer.count_tokens(doc.text) * self.config.retention_ratio))
                          if self.config.summary_length_policy == "ratio" else self.config.summary_max_new_tokens)
                requested = target if self.config.summary_length_policy == "ratio" else None
                produce = lambda budget: backend.generate(prompts.summary_messages(doc.text, method,
                    budget if requested is not None else None), budget)
                attempts = []
                if self.config.budget_policy == "reference_cap":
                    generated, seconds, attempts = fit_reference_cap(produce, self.tokenizer.count_tokens, target,
                        target, self.config.budget_attempt_limit, integer_control=True)
                else:
                    generated = produce(target)
                    seconds = generated.seconds
                self._save_context(self._context_key(method, doc), method, doc, generated.text, seconds,
                                   generation=asdict(generated), budget_attempts=attempts,
                                   requested_summary_tokens=requested, summary_generation_cap=target)
                print(f"summary {i}/{len(pending)}: {method}/{doc.id} [{seconds:.2f}s]", flush=True)
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
        backend = self._load_backend("compressor", self.compressor_factory, self.config.compressor)
        try:
            self._identity("compressor", backend.identity)
            compress = lambda text, hypothesis, rate: (backend.compress(text, hypothesis, rate, budget_mode="rate")
                if self.config.compressor_budget_mode == "rate" else backend.compress(text, hypothesis, rate))
            self._warmup("compressor", lambda: compress(
                "The recipient must protect confidential records. Legally required disclosure is permitted.\n" * 8,
                "Legally required disclosure is permitted.", self.config.retention_ratio))
            for i, (doc, example) in enumerate(pending, 1):
                attempts = []
                if self.config.budget_policy == "reference_cap":
                    cap = max(1, int(self.tokenizer.count_tokens(doc.text) * self.config.retention_ratio))
                    compressed, seconds, attempts = fit_reference_cap(
                        lambda rate: compress(doc.text, example.hypothesis, rate), self.tokenizer.count_tokens,
                        cap, self.config.retention_ratio, self.config.budget_attempt_limit)
                else:
                    compressed = compress(doc.text, example.hypothesis, self.config.retention_ratio)
                    seconds = compressed.seconds
                self._save_context(self._context_key("longllmlingua", doc, example), "longllmlingua", doc,
                                   compressed.text, seconds, hypothesis_id=example.hypothesis_id,
                                   scorer_tokens=compressed.scorer_tokens,
                                   target_scorer_tokens=compressed.target_scorer_tokens,
                                   prefix_fallbacks=compressed.prefix_fallbacks, budget_attempts=attempts,
                                   compressor_budget_mode=self.config.compressor_budget_mode,
                                   requested_compressor_rate=self.config.retention_ratio)
                print(f"compression {i}/{len(pending)}: {example.key} [{seconds:.2f}s]", flush=True)
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
        if self.config.shuffle_judgments:
            work.sort(key=lambda item: digest({"seed": self.config.seed, "key": item[3]}))
        self._ensure_tokenizer()
        backend = self._load_backend("judge", self.backend_factory, self.config.judge)
        try:
            self._identity("judge", backend.identity)
            generate = backend.generate_labels if self.config.judge_decoding == "constrained" else backend.generate
            self._warmup("judge", lambda: generate(prompts.judge_messages(
                "The recipient must protect confidential records.", "The recipient must protect confidential records."),
                self.config.judge_max_new_tokens))
            for i, (method, doc, example, key, context_key) in enumerate(work, 1):
                context = doc.text if method == "original" else self.contexts.rows[context_key]["text"]
                generated = generate(prompts.judge_messages(context, example.hypothesis), self.config.judge_max_new_tokens)
                prediction = prompts.parse_label(generated.text)
                if self.config.judge_decoding == "constrained" and (prediction is None or generated.finish_reason != "eos"):
                    raise ValueError("Constrained judgment violated the label-only protocol; no prediction saved")
                self.predictions.add(key, {
                    "condition": method, "example_id": example.key, "document_id": doc.id,
                    "hypothesis_id": example.hypothesis_id, "hypothesis": example.hypothesis,
                    "gold": example.gold, "prediction": prediction,
                    "judge_decoding": self.config.judge_decoding,
                    "gold_evidence": list(example.evidence), "has_gold_evidence": bool(example.evidence),
                    "context_key": context_key if method != "original" else None,
                    "context_fingerprint": digest({"text": context}),
                    "context_tokens": self.tokenizer.count_tokens(context),
                    "original_tokens": self.tokenizer.count_tokens(doc.text),
                    "generation": asdict(generated),
                })
                print(f"judgment {i}/{len(work)}: {key} [{generated.seconds:.2f}s]", flush=True)
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
            lengths = [r["context_tokens"] / r["original_tokens"] for r in predictions]
            latencies = [r["generation"]["seconds"] for r in predictions]
            result.update({
                "compression_calls": len(contexts), "compression_seconds": compression_seconds,
                "judgment_seconds": judgment_seconds, "total_inference_seconds": total_seconds,
                "amortized_seconds_per_pair": total_seconds / len(predictions),
                "mean_actual_token_retention": mean(r["context_tokens"] / r["original_tokens"] for r in predictions),
                "corpus_token_retention": sum(r["context_tokens"] for r in predictions) / sum(r["original_tokens"] for r in predictions),
                "median_actual_token_retention": quantile(lengths, .5),
                "p10_actual_token_retention": quantile(lengths, .1),
                "p90_actual_token_retention": quantile(lengths, .9),
                "judgment_median_seconds": quantile(latencies, .5),
                "judgment_p95_seconds": quantile(latencies, .95),
                "judgment_length_capped_calls": sum(r["generation"]["finish_reason"] == "length" for r in predictions),
                "judgment_prompt_tokens": sum(r["generation"]["prompt_tokens"] for r in predictions),
                "judgment_output_tokens": sum(r["generation"]["output_tokens"] for r in predictions),
                "invalid_output_rate": result["invalid_outputs"] / len(predictions),
                "over_target_context_calls": (None if any(c["target_reference_tokens"] is None for c in contexts) else
                                               sum(c["context_tokens"] > c["target_reference_tokens"] for c in contexts)),
                "length_capped_summary_calls": sum(c.get("generation", {}).get("finish_reason") == "length" for c in contexts),
                "first_line_sensitivity": classification([{**r, "prediction": prompts.parse_first_line(r["generation"]["text"])}
                                                          for r in predictions]),
            })
            if method != "original" and "original" in grouped:
                result.update(paired_metrics(grouped["original"], predictions))
            metrics[method] = result
        report = {"schema_version": 2, "complete": True, "signature": self.manifest["signature"],
                  "dataset": inspect(self.config, self.dataset), "metrics": metrics,
                  "protocol": {"judge_decoding": self.config.judge_decoding, "budget_policy": self.config.budget_policy,
                               "summary_length_policy": self.config.summary_length_policy,
                               "compressor_budget_mode": self.config.compressor_budget_mode,
                               "warmup_calls_per_model_load": self.config.warmup_calls,
                               "judgment_order": "seeded hash interleaving" if self.config.shuffle_judgments else "condition blocks",
                               "length_matched": False, "timing_is_repeated_benchmark": False},
                  "notes": ["Three gold classes retained, including NotMentioned.",
                            "No semantic gold-evidence preservation score is computed in the base experiment.",
                            "Lengths use the judge tokenizer; native compressor rate and summary generation cap are distinct, not a common output length.",
                            "Inference time includes per-call tokenization and compression/generation, excludes model loading and checkpoint I/O.",
                            "All budget-fitting inference attempts count toward compression time when reference_cap is enabled.",
                            "Warmup calls are logged separately and excluded; measurement is single-pass operational latency, not a repeated benchmark.",
                            "Native settings compare method behavior at unequal achieved lengths, not a length-controlled causal effect.",
                            "LongLLMLingua rate uses the upstream native-token formula including question/chunk accounting; it is not an exact judge-token ratio.",
                            "Repeated runs reuse stored timings; they are not new latency measurements."]}
        write_json(self.directory / "metrics.json", report)
        with (self.directory / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            fields = ["condition", "n", "accuracy", "macro_f1", "decision_flip_rate", "decision_agreement_rate",
                      "valid_pair_n", "valid_pair_coverage", "correct_decision_retention", "correct_to_wrong_n",
                      "wrong_to_correct_n", "mean_actual_token_retention", "corpus_token_retention",
                      "compression_seconds", "judgment_seconds", "total_inference_seconds", "amortized_seconds_per_pair",
                      "judgment_median_seconds", "judgment_p95_seconds", "length_capped_summary_calls",
                      "invalid_outputs", "invalid_output_rate"]
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
            invocation = {"stage": stage, "contexts_before": len(self.contexts.rows),
                          "predictions_before": len(self.predictions.rows), "status": "running"}
            self.manifest.setdefault("invocations", []).append(invocation)
            write_json(self.manifest_path, self.manifest)
            try:
                for name in ("summarize", "compress", "judge", "report"):
                    if stage in ("all", name):
                        stage_started = time.perf_counter()
                        invocation["active_stage"] = name
                        result = getattr(self, name)()
                        self.manifest["stages"][name] = {"last_invocation_wall_seconds": time.perf_counter() - stage_started,
                                                        "completed": True}
                        write_json(self.manifest_path, self.manifest)
                invocation["status"] = "completed"
                self.manifest["stages"][stage] = {"last_invocation_wall_seconds": time.perf_counter() - started,
                                                  "completed": True}
            except BaseException as exc:
                invocation["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                invocation["error"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                invocation.update({"wall_seconds": time.perf_counter() - started,
                                   "contexts_after": len(self.contexts.rows), "predictions_after": len(self.predictions.rows)})
                write_json(self.manifest_path, self.manifest)
            return result
