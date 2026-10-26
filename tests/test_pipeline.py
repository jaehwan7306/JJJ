import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from contractnli.backend import Generation
from contractnli.compression import Compression, split_context
from contractnli.config import Config, ModelConfig, load_config
from contractnli.data import load_dataset
from contractnli.experiment import Experiment, inspect
from contractnli.metrics import classification, paired_metrics
from contractnli.prompts import judge_messages, parse_label, summary_messages
from contractnli.storage import JsonlStore


def fixture():
    text = "The recipient must return records, except legal archive copies."
    return {"labels": {
        "h1": {"hypothesis": "The recipient must return records."},
        "h2": {"hypothesis": "No archive copies may be kept."},
        "h3": {"hypothesis": "The recipient may hire consultants."}},
        "documents": [{"id": 7, "text": text, "spans": [[0, len(text)]],
                       "annotation_sets": [{"annotations": {
                           "h1": {"choice": "Entailment", "spans": [0]},
                           "h2": {"choice": "Contradiction", "spans": [0]},
                           "h3": {"choice": "NotMentioned", "spans": []}}}]}]}


class FakeTokenizer:
    identity = {"test_only_tokenizer": "characters"}

    def __init__(self, config):
        pass

    def count_tokens(self, text):
        return len(text)


class FakeBackend:
    calls = []
    closed = 0

    def __init__(self, config, seed):
        self.identity = {"test_only_model": config.name}

    def generate(self, messages, max_new_tokens):
        self.calls.append((messages, max_new_tokens))
        if "Summarize" in messages[0]["content"]:
            text = "Short contract."
        elif "No archive copies" in messages[1]["content"]:
            text = "Contradiction"
        elif "hire consultants" in messages[1]["content"]:
            text = "NotMentioned"
        else:
            text = "Entailment"
        return Generation(text, 100, min(len(text), max_new_tokens), 0.25, "eos")

    def close(self):
        type(self).closed += 1


class FakeCompressor:
    calls = []

    def __init__(self, config, seed):
        self.identity = {"test_only_compressor": config.name}

    def compress(self, text, hypothesis, retention_ratio):
        self.calls.append((text, hypothesis, retention_ratio))
        return Compression("Short context.", 0.5, 14, int(len(text) * retention_ratio))

    def close(self):
        pass


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.dataset_path = self.directory / "test.json"
        self.dataset_path.write_text(json.dumps(fixture()), encoding="utf-8")
        self.config = Config(dataset=str(self.dataset_path), output_dir=str(self.directory / "run"))
        FakeBackend.calls = []
        FakeBackend.closed = 0
        FakeCompressor.calls = []

    def experiment(self, config=None, backend=FakeBackend):
        return Experiment(config or self.config, backend, FakeCompressor, FakeTokenizer)

    def execute(self, stage="all", config=None, backend=FakeBackend):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.experiment(config, backend).run(stage)

    def test_all_three_labels_and_absent_evidence_are_retained(self):
        dataset = load_dataset(self.dataset_path)
        self.assertEqual(len(dataset.examples), 3)
        self.assertEqual(dataset.not_mentioned_pairs, 1)
        self.assertEqual(dataset.examples[-1].gold, "NotMentioned")
        self.assertEqual(dataset.examples[-1].evidence, ())
        self.assertEqual(dataset.examples[0].evidence[0], fixture()["documents"][0]["text"])

    def test_bad_evidence_fails_instead_of_slicing_wrong_text(self):
        content = fixture()
        content["documents"][0]["spans"] = [[0, 999999]]
        self.dataset_path.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "offsets"):
            load_dataset(self.dataset_path)

    def test_multiple_annotation_sets_need_explicit_policy(self):
        content = fixture()
        content["documents"][0]["annotation_sets"] *= 2
        self.dataset_path.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "aggregation"):
            load_dataset(self.dataset_path)

    def test_zero_subset_rejected(self):
        with self.assertRaises(ValueError):
            load_dataset(self.dataset_path, 0)

    def test_dry_inspection_counts_document_and_question_requests(self):
        summary = inspect(self.config, load_dataset(self.dataset_path))
        self.assertEqual(summary["summary_requests"], 2)
        self.assertEqual(summary["question_aware_compression_requests"], 3)
        self.assertEqual(summary["judgment_requests"], 12)

    def test_full_pipeline_reuses_document_summaries_keeps_all_pairs(self):
        result = self.execute()
        self.assertTrue(result["complete"])
        self.assertEqual(len(FakeBackend.calls), 14)
        self.assertEqual(len(FakeCompressor.calls), 3)
        self.assertEqual(FakeBackend.closed, 2)
        contexts = JsonlStore(Path(self.config.output_dir) / "contexts.jsonl")
        self.assertEqual(len(contexts.rows), 5)
        for method, metric in result["metrics"].items():
            self.assertEqual(metric["n"], 3)
            self.assertEqual(metric["accuracy"], 1)
            self.assertEqual(metric["macro_f1"], 1)
            if method != "original":
                self.assertEqual(metric["decision_flip_rate"], 0)
        self.assertEqual(result["metrics"]["naive"]["compression_calls"], 1)
        self.assertEqual(result["metrics"]["longllmlingua"]["compression_calls"], 3)
        self.assertAlmostEqual(result["metrics"]["naive"]["total_inference_seconds"], 1.0)
        self.assertAlmostEqual(result["metrics"]["longllmlingua"]["total_inference_seconds"], 2.25)

    def test_resume_does_not_generate_again(self):
        self.execute()
        previous_calls = len(FakeBackend.calls)
        self.execute()
        self.assertEqual(len(FakeBackend.calls), previous_calls)
        self.assertEqual(len(FakeCompressor.calls), 3)

    def test_single_gpu_never_has_two_models_loaded(self):
        active = set()
        events = []

        class SingleGpuModel(FakeBackend):
            def __init__(self, config, seed):
                if active:
                    raise AssertionError(f"Two models loaded: {active} and {config.name}")
                super().__init__(config, seed)
                self.name = config.name
                active.add(self.name)
                events.append(("load", self.name))

            def close(self):
                active.remove(self.name)
                events.append(("unload", self.name))

        class SingleGpuCompressor(FakeCompressor):
            def __init__(self, config, seed):
                if active:
                    raise AssertionError(f"Two models loaded: {active} and {config.name}")
                super().__init__(config, seed)
                self.name = config.name
                active.add(self.name)
                events.append(("load", self.name))

            def close(self):
                active.remove(self.name)
                events.append(("unload", self.name))

        with contextlib.redirect_stdout(io.StringIO()):
            result = Experiment(self.config, SingleGpuModel, SingleGpuCompressor, FakeTokenizer).run()
        self.assertTrue(result["complete"])
        names = [self.config.summarizer.name, self.config.compressor.name, self.config.judge.name]
        self.assertEqual(events, [(action, name) for name in names for action in ("load", "unload")])
        self.assertFalse(active)

    def test_stage_handoff(self):
        self.execute("summarize")
        with self.assertRaisesRegex(ValueError, "Missing longllmlingua"):
            self.execute("judge")
        self.execute("compress")
        self.execute("judge")
        self.assertTrue(self.execute("report")["complete"])

    def test_changed_config_rejects_stale_results(self):
        self.execute()
        with self.assertRaisesRegex(ValueError, "changed"):
            self.execute(config=replace(self.config, retention_ratio=0.5))

    def test_changed_dataset_rejects_stale_results(self):
        self.execute()
        self.dataset_path.write_text(json.dumps(fixture(), indent=2))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.execute()

    def test_incomplete_run_cannot_emit_metrics(self):
        self.execute("summarize")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.execute("report")
        self.assertFalse((Path(self.config.output_dir) / "metrics.json").exists())

    def test_generation_error_releases_model_and_preserves_checkpoint(self):
        class BrokenBackend(FakeBackend):
            def generate(self, messages, max_new_tokens):
                if len(self.calls) >= 1:
                    raise RuntimeError("simulated out of memory")
                return super().generate(messages, max_new_tokens)
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.execute("summarize", backend=BrokenBackend)
        self.assertEqual(len(JsonlStore(Path(self.config.output_dir) / "contexts.jsonl").rows), 1)
        self.assertFalse((Path(self.config.output_dir) / ".run.lock").exists())
        self.execute()

    def test_reference_budget_uses_original_length_and_ratio(self):
        self.execute("summarize")
        contexts = JsonlStore(Path(self.config.output_dir) / "contexts.jsonl")
        target = int(len(fixture()["documents"][0]["text"]) * 0.3)
        for context in contexts.rows.values():
            self.assertEqual(context["target_reference_tokens"], target)
        self.assertTrue(all(budget == target for _, budget in FakeBackend.calls))


class MetricTests(unittest.TestCase):
    def rows(self, predictions):
        return [{"example_id": str(i), "gold": gold, "prediction": pred}
                for i, (gold, pred) in enumerate(zip(("Entailment", "Contradiction", "NotMentioned"), predictions))]

    def test_three_class_f1_counts_invalid_outputs_as_errors(self):
        result = classification(self.rows(["Entailment", None, "NotMentioned"]))
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(result["invalid_outputs"], 1)

    def test_flip_agreement_and_correct_retention_are_distinct(self):
        original = self.rows(["Entailment", "Entailment", "NotMentioned"])
        compressed = self.rows(["Contradiction", "Contradiction", None])
        result = paired_metrics(original, compressed)
        self.assertEqual(result["valid_pair_n"], 2)
        self.assertEqual(result["decision_flip_rate"], 1)
        self.assertEqual(result["correct_decision_retention"], 0)
        self.assertEqual(result["correct_to_wrong_n"], 2)
        self.assertEqual(result["wrong_to_correct_n"], 1)

    def test_pairing_rejects_missing_examples(self):
        rows = self.rows(["Entailment", "Contradiction", "NotMentioned"])
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_metrics(rows, rows[:2])

    def test_no_predictions_is_not_success(self):
        with self.assertRaises(ValueError):
            classification([])


class PromptAndStorageTests(unittest.TestCase):
    def test_summary_prompts_have_no_question_or_gold(self):
        naive = summary_messages("contract text", "naive", 50)
        contextualized = summary_messages("contract text", "contextualized", 50)
        self.assertNotEqual(naive, contextualized)
        self.assertNotIn("hypothesis", str(contextualized))
        self.assertNotIn("gold", str(contextualized))
        self.assertIn("conditions", contextualized[0]["content"])

    def test_judge_prompts_keep_three_labels(self):
        prompt = judge_messages("source", "claim")
        self.assertIn("NotMentioned", prompt[0]["content"])

    def test_invalid_labels_not_guessed(self):
        self.assertEqual(parse_label(" Entailment\n"), "Entailment")
        self.assertIsNone(parse_label("Entailment or Contradiction"))
        self.assertIsNone(parse_label("The answer is Entailment"))

    def test_chunking_preserves_all_characters(self):
        text = "first line\n" + "long content " * 100 + "\nlast line"
        chunks = split_context(text, len, limit=50)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(c) <= 50 for c in chunks))

    def test_partial_jsonl_is_reported_instead_of_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"key":"ok"}\n{"key":')
            with self.assertRaisesRegex(ValueError, "Invalid checkpoint"):
                JsonlStore(path)

    def test_default_config_matches_requested_four_conditions(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/base.toml")
        self.assertEqual(len(config.conditions), 4)
        self.assertEqual(config.retention_ratio, 0.3)
        self.assertEqual(config.summarizer.name, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(config.judge.name, "Qwen/Qwen3-8B")
        self.assertEqual(config.compressor.name, "NousResearch/Llama-2-7b-hf")

    def test_invalid_quantization_device_combination_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            ModelConfig("test", device="cpu", dtype="float32", quantization="4bit")


if __name__ == "__main__":
    unittest.main()
