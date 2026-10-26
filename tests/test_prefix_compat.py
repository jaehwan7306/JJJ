import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prefix_compat_test_module", ROOT / "tools/llmlingua_prefix_compat.py")
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)
HAS_LINGUA = all(importlib.util.find_spec(name) for name in ("llmlingua", "transformers", "tokenizers", "torch"))


class BoundaryValidationTests(unittest.TestCase):
    def test_cross_boundary_token_is_rejected(self):
        class Tokenizer:
            is_fast = True

            def __call__(self, text, **kwargs):
                return {"input_ids": [1, 2], "offset_mapping": [(0, 2), (2, 6)]}

        with self.assertRaisesRegex(ValueError, "crosses"):
            compat.prefix_boundary_from_offsets(Tokenizer(), "abc", "def")

    def test_slow_tokenizer_is_rejected(self):
        class SlowTokenizer:
            is_fast = False

        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            compat.prefix_boundary_from_offsets(SlowTokenizer(), "prefix", "context")


@unittest.skipUnless(HAS_LINGUA, "optional inference dependencies not installed")
class PrefixCompatibilityTests(unittest.TestCase):
    def setUp(self):
        from llmlingua import PromptCompressor
        from contractnli.experiment import Experiment

        self.original_prefix = PromptCompressor.get_prefix_length
        self.original_open = Experiment._open
        self.addCleanup(setattr, PromptCompressor, "get_prefix_length", self.original_prefix)
        self.addCleanup(setattr, Experiment, "_open", self.original_open)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def tokenizer(self):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.trainers import BpeTrainer
        from transformers import PreTrainedTokenizerFast

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer.train_from_iterator([
            "Decide the hypothesis\n\n",
            "The recipient must return records . Legal archive copies may stay .\n",
            "Determine whether the following hypothesis is entailed, contradicted, or not mentioned: ",
        ], BpeTrainer(vocab_size=512, special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
                      initial_alphabet=ByteLevel.alphabet(), show_progress=False))
        return PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>",
                                      unk_token="<unk>", pad_token="<pad>", clean_up_tokenization_spaces=True)

    def compressor(self, tokenizer):
        from llmlingua import PromptCompressor

        compressor = PromptCompressor.__new__(PromptCompressor)
        compressor.tokenizer = tokenizer
        return compressor

    def test_reproduces_round_trip_assertion_then_verifies_boundary(self):
        tokenizer = self.tokenizer()
        compressor = self.compressor(tokenizer)
        prefix = "Decide the hypothesis\n\n"
        text = "The recipient must return records . Legal archive copies may stay ."
        with self.assertRaises(AssertionError):
            self.original_prefix(compressor, prefix, text)
        compat.apply_prefix_fix()
        index = compressor.get_prefix_length(prefix, text)
        encoded = tokenizer(prefix + text, add_special_tokens=False, return_offsets_mapping=True)
        self.assertTrue(all(end <= len(prefix) for _, end in encoded["offset_mapping"][:index]))
        self.assertTrue(all(start >= len(prefix) for start, _ in encoded["offset_mapping"][index:]))
        self.assertEqual(tokenizer.decode(encoded["input_ids"][index:], clean_up_tokenization_spaces=False), text)

    def test_original_successful_boundary_is_unchanged(self):
        compressor = self.compressor(self.tokenizer())
        prefix, text = "Decide the hypothesis\n\n", "The recipient must return records"
        before = self.original_prefix(compressor, prefix, text)
        compat.apply_prefix_fix()
        with patch.object(compat, "prefix_boundary_from_offsets", side_effect=AssertionError("Must use original result")):
            self.assertEqual(compressor.get_prefix_length(prefix, text), before)

    def test_reapplying_does_not_stack_adapters(self):
        from llmlingua import PromptCompressor

        first = compat.apply_prefix_fix()
        method = PromptCompressor.get_prefix_length
        self.assertEqual(compat.apply_prefix_fix(), first)
        self.assertIs(PromptCompressor.get_prefix_length, method)

    def test_existing_checkpoints_resume_without_recomputation_or_signature_change(self):
        from contractnli.config import Config
        from contractnli.experiment import Experiment
        from test_pipeline import FakeBackend, FakeCompressor, FakeTokenizer, fixture

        dataset_path = self.directory / "test.json"
        dataset_path.write_text(json.dumps(fixture()), encoding="utf-8")
        output = self.directory / "run"
        config = Config(dataset=str(dataset_path), output_dir=str(output))
        FakeBackend.calls = []
        with contextlib.redirect_stdout(io.StringIO()):
            first = Experiment(config, FakeBackend, FakeCompressor, FakeTokenizer).run()
        old_contexts = (output / "contexts.jsonl").read_bytes()
        old_predictions = (output / "predictions.jsonl").read_bytes()
        old_calls = len(FakeBackend.calls)
        compat.apply_prefix_fix()
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = Experiment(config, FakeBackend, FakeCompressor, FakeTokenizer).run()
        self.assertEqual(first["signature"], resumed["signature"])
        self.assertEqual(first["metrics"], resumed["metrics"])
        self.assertEqual(old_contexts, (output / "contexts.jsonl").read_bytes())
        self.assertEqual(old_predictions, (output / "predictions.jsonl").read_bytes())
        self.assertEqual(old_calls, len(FakeBackend.calls))
        manifest = json.loads((output / "run.json").read_text())
        record = manifest["compatibility_patches"][compat.FIX_ID]
        self.assertEqual(record["compressed_contexts_before_registration"], 3)
        self.assertEqual(record["specification"], compat.apply_prefix_fix())
        # Integrity is enforced for the separately versioned compatibility adapter.
        record["specification"]["implementation_sha256"] = "different"
        (output / "run.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "adapter hash differs"):
            Experiment(config, FakeBackend, FakeCompressor, FakeTokenizer).run("report")

    def test_real_llmlingua_compression_with_non_round_trip_text(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from contractnli.compression import LongLLMLinguaBackend
        from contractnli.config import ModelConfig

        torch.set_num_threads(1)
        tokenizer = self.tokenizer()
        config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=512, bos_token_id=1, eos_token_id=2, pad_token_id=3)
        torch.manual_seed(42)
        LlamaForCausalLM(config).save_pretrained(self.directory)
        tokenizer.save_pretrained(self.directory)
        model_config = ModelConfig(str(self.directory), device="cpu", dtype="float32", quantization="none",
                                   max_input_tokens=512)
        # The exact-case precheck also uses the cached tokenizer without weights.
        from contractnli.config import Config
        from test_pipeline import fixture
        dataset_path = self.directory / "test.json"
        dataset_path.write_text(json.dumps(fixture()), encoding="utf-8")
        check = compat.validate_prefix_case(Config(dataset=str(dataset_path), compressor=model_config), "7", "h1")
        self.assertEqual(check["status"], "PASS")
        self.assertFalse(check["model_weights_loaded"])
        class DisplayTokenizer:
            def encode(self, text):
                return text.split()
        compat.apply_prefix_fix()
        with patch("llmlingua.prompt_compressor.tiktoken.encoding_for_model", return_value=DisplayTokenizer()):
            backend = LongLLMLinguaBackend(model_config, 42)
        try:
            text = "The recipient must return records . Legal archive copies may stay .\n" * 40
            result = backend.compress(text, "The recipient must return records", 0.3)
            self.assertTrue(result.text)
            self.assertLess(result.scorer_tokens, backend.backend.count_tokens(text))
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
