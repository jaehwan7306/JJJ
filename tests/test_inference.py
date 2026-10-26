"""Offline integration tests with tiny RANDOM models, not research results."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_INFERENCE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "tokenizers"))
HAS_LINGUA = importlib.util.find_spec("llmlingua") is not None


@unittest.skipUnless(HAS_INFERENCE, "optional inference dependencies not installed")
class InferenceIntegrationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.trainers import BpeTrainer
        from transformers import PreTrainedTokenizerFast

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.torch = torch
        torch.set_num_threads(1)
        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()
        corpus = ["The recipient must return records except legal archive copies\n",
                  "Determine whether the following hypothesis is entailed, contradicted, or not mentioned: ",
                  "The recipient may keep legal archive copies", "The contract is confidential",
                  "Entailment Contradiction NotMentioned system user assistant"]
        tok.train_from_iterator(corpus, BpeTrainer(vocab_size=512, special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
                                                  initial_alphabet=ByteLevel.alphabet(), show_progress=False))
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
                                                unk_token="<unk>", pad_token="<pad>")
        self.tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"

    def save_llama(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        self.torch.manual_seed(42)
        config = LlamaConfig(vocab_size=len(self.tokenizer), hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=512, bos_token_id=1, eos_token_id=2, pad_token_id=3)
        model = LlamaForCausalLM(config).eval()
        model.save_pretrained(self.path)
        self.tokenizer.save_pretrained(self.path)
        return model

    def test_tiny_qwen3_real_generate_and_overflow_rejection(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from contractnli.backend import TransformersBackend
        from contractnli.config import ModelConfig

        config = Qwen3Config(vocab_size=len(self.tokenizer), hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=128,
                             bos_token_id=1, eos_token_id=2, pad_token_id=3)
        Qwen3ForCausalLM(config).save_pretrained(self.path)
        self.tokenizer.save_pretrained(self.path)
        model_config = ModelConfig(str(self.path), device="cpu", dtype="float32", quantization="none",
                                   max_input_tokens=128)
        backend = TransformersBackend(model_config, 42)
        try:
            result = backend.generate([{"role": "user", "content": "The contract is confidential"}], 4)
            self.assertGreater(result.output_tokens, 0)
            self.assertLessEqual(result.output_tokens, 4)
            self.assertGreater(result.seconds, 0)
            with self.assertRaisesRegex(ValueError, "No truncation"):
                backend.generate([{"role": "user", "content": "contract " * 130}], 4)
        finally:
            backend.close()

    def test_cache_adapter_preserves_incremental_logits(self):
        from transformers.cache_utils import DynamicCache
        from contractnli.compression import LegacyCacheAdapter

        model = self.save_llama()
        adapter = LegacyCacheAdapter(model)
        ids = self.torch.tensor([[10, 11, 12, 13]])
        with self.torch.inference_mode():
            first = adapter(ids, use_cache=True)
            self.assertIsInstance(first.past_key_values, tuple)
            legacy = first.past_key_values
            native_cache = DynamicCache.from_legacy_cache(tuple(tuple(t.clone() for t in layer) for layer in legacy))
            new_ids = self.torch.tensor([[14, 15]])
            native = model(new_ids, past_key_values=native_cache, use_cache=True)
            bridged = adapter(new_ids, past_key_values=[list(layer) for layer in legacy], use_cache=True)
            self.torch.testing.assert_close(native.logits, bridged.logits)
            self.assertEqual(bridged.past_key_values[0][0].shape[-2], 6)

    def test_preflight_detects_full_original_context_overflow(self):
        from contractnli.config import Config, ModelConfig
        from contractnli.preflight import preflight

        self.save_llama()
        path = self.path / "test.json"
        path.write_text(json.dumps({"labels": {"h1": {"hypothesis": "The recipient must return records"}},
                                    "documents": [{"id": 1, "text": "The recipient must return records " * 10,
                                                   "spans": [], "annotation_sets": [{"annotations": {
                                                       "h1": {"choice": "NotMentioned", "spans": []}}}]}]}))
        judge = ModelConfig(str(self.path), device="cpu", dtype="float32", quantization="none", max_input_tokens=16)
        config = Config(dataset=str(path), conditions=("original",), judge=judge)
        result = preflight(config)
        self.assertFalse(result["ok"])
        self.assertEqual(result["context_overflows"][0]["example_id"], "1/h1")
        self.assertGreater(result["maxima"]["original_judge_prompt_plus_output_tokens"], 16)

    @unittest.skipUnless(HAS_LINGUA, "optional llmlingua not installed")
    def test_actual_longllmlingua_compression_with_tiny_random_scorer(self):
        from contractnli.compression import LongLLMLinguaBackend
        from contractnli.config import ModelConfig

        self.save_llama()
        model_config = ModelConfig(str(self.path), device="cpu", dtype="float32", quantization="none",
                                   max_input_tokens=512)
        # tiktoken is used only for the upstream display statistics. Avoid network
        # in the test; all ranking/compression still uses the real local scorer.
        class DisplayTokenizer:
            def encode(self, text):
                return text.split()
        with patch("llmlingua.prompt_compressor.tiktoken.encoding_for_model", return_value=DisplayTokenizer()):
            backend = LongLLMLinguaBackend(model_config, 42)
        try:
            text = "The recipient must return records except legal archive copies\n" * 70
            result = backend.compress(text, "The recipient may keep legal archive copies", 0.3)
            self.assertTrue(result.text)
            self.assertGreater(result.seconds, 0)
            self.assertLess(result.scorer_tokens, backend.backend.count_tokens(text))
            self.assertEqual(result.target_scorer_tokens, int(backend.backend.count_tokens(text) * 0.3))
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
