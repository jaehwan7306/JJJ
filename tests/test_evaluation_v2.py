import contextlib
import io
import json
import tempfile
import unittest
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from contractnli.backend import Generation, LabelConstraint
from contractnli.budget import fit_reference_cap
from contractnli.config import Config, ModelConfig, load_config
from contractnli.experiment import Experiment, inspect
from contractnli.prompts import parse_first_line, summary_messages
from contractnli.storage import JsonlStore
from test_pipeline import FakeBackend, FakeCompressor, FakeTokenizer, fixture


class ConstraintTests(unittest.TestCase):
    def tokenizer(self):
        class CharTokenizer:
            def encode(self, text, **kwargs):
                return list(text.encode())

            def decode(self, ids, **kwargs):
                return bytes(ids).decode()
        return CharTokenizer()

    def test_trie_allows_only_whole_labels_then_eos(self):
        from contractnli.data import LABELS
        class IDs(list):
            def __getitem__(self, item):
                result = super().__getitem__(item)
                return IDs(result) if isinstance(result, list) else result
            def tolist(self):
                return list(self)
        c = LabelConstraint(self.tokenizer(), 2, 0, 32)
        self.assertEqual(c(0, IDs([9, 9])), sorted(ord(l[0]) for l in LABELS))
        for label in LABELS:
            prefix = IDs([9, 9])
            for token in label.encode():
                self.assertIn(token, c(0, prefix))
                self.assertNotIn(0, c(0, prefix))
                prefix.append(token)
            self.assertEqual(c(0, prefix), [0])
        with self.assertRaises(RuntimeError):
            c(0, IDs([9, 9, ord('X')]))

    def test_too_small_budget_fails_before_generation(self):
        with self.assertRaisesRegex(ValueError, 'complete label plus EOS'):
            LabelConstraint(self.tokenizer(), 0, 0, 2)

    def test_secondary_parser_is_anchored_and_does_not_guess(self):
        self.assertEqual(parse_first_line('**Entailment**\n\nExplanation'), 'Entailment')
        self.assertEqual(parse_first_line(' NotMentioned\nexplanation'), 'NotMentioned')
        for text in ['', 'The answer is Entailment', 'Entailment or Contradiction', '**Entailment** or NotMentioned']:
            self.assertIsNone(parse_first_line(text))

    def test_fixed_summary_prompt_has_no_length_instruction(self):
        for m in ('naive', 'contextualized'):
            p = summary_messages('contract', m, None)
            self.assertNotIn('tokens', p[0]['content'])
            self.assertNotIn('None', p[0]['content'])
            self.assertIn('only the summary', p[0]['content'])


class NewPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'test.json'
        self.dataset.write_text(json.dumps(fixture()))
        self.config = Config(dataset=str(self.dataset), output_dir=str(self.root / 'run'),
            judge_decoding='constrained', summary_length_policy='fixed', summary_max_new_tokens=2048,
            retention_ratio=.55, compressor_budget_mode='rate', warmup_calls=1, shuffle_judgments=True)
        FakeBackend.calls, FakeBackend.closed, FakeCompressor.calls = [], 0, []

    def run_experiment(self, config=None, backend=FakeBackend):
        with contextlib.redirect_stdout(io.StringIO()):
            return Experiment(config or self.config, backend, FakeCompressor, FakeTokenizer).run()

    def test_native_settings_warmup_exclusion_and_resume(self):
        report = self.run_experiment()
        self.assertEqual(report['metrics']['naive']['compression_seconds'], .25)
        self.assertEqual(report['metrics']['longllmlingua']['compression_seconds'], 1.5)
        self.assertEqual(report['metrics']['contextualized']['invalid_outputs'], 0)
        summaries = [call for call in FakeBackend.calls if 'Summarize' in call[0][0]['content'] and call[1] == 2048]
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all('tokens' not in msgs[0]['content'] for msgs, _ in summaries))
        manifest = json.loads((self.root / 'run/run.json').read_text())
        self.assertEqual(len(manifest['warmups']), 3)
        self.assertEqual(manifest['invocations'][0]['status'], 'completed')
        before = len(FakeBackend.calls)
        self.assertEqual(report['metrics'], self.run_experiment()['metrics'])
        self.assertEqual(len(FakeBackend.calls), before)
        summary = inspect(self.config, Experiment(self.config).dataset)
        self.assertIsNone(summary['target_token_retention'])
        self.assertEqual(summary['compressor_rate'], .55)
        self.assertFalse(report['protocol']['length_matched'])

    def test_judgment_order_is_interleaved_and_stable_on_resume(self):
        self.run_experiment()
        p = self.root / 'run/predictions.jsonl'
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        self.assertNotEqual([r['condition'] for r in rows], sorted(r['condition'] for r in rows))
        p.write_text('\n'.join(json.dumps(r) for r in rows[:4]) + '\n')
        self.run_experiment()
        self.assertEqual(rows, [json.loads(l) for l in p.read_text().splitlines()])

    def test_bad_constrained_output_stops_and_records_failure(self):
        class Broken(FakeBackend):
            def generate_labels(self, messages, max_new_tokens):
                return Generation('Entailment\nextra text', 100, 16, .25, 'length')
        with self.assertRaisesRegex(ValueError, 'violated'):
            self.run_experiment(backend=Broken)
        self.assertEqual(len(JsonlStore(self.root / 'run/predictions.jsonl').rows), 0)
        self.assertFalse((self.root / 'run/.run.lock').exists())
        m = json.loads((self.root / 'run/run.json').read_text())
        self.assertEqual(m['invocations'][-1]['status'], 'failed')

    def test_corrupted_cached_gold_cannot_silently_change_score(self):
        self.run_experiment()
        p = self.root / 'run/predictions.jsonl'
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        rows[0]['gold'] = 'Contradiction' if rows[0]['gold'] != 'Contradiction' else 'Entailment'
        p.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            self.run_experiment()

    def test_changed_context_text_is_detected_before_inference(self):
        self.run_experiment()
        p = self.root / 'run/contexts.jsonl'
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        rows[0]['text'] += ' altered'
        p.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        with self.assertRaisesRegex(ValueError, 'Invalid context'):
            self.run_experiment()

    def test_cross_role_hardware_mismatch_is_rejected(self):
        class GPUBackend(FakeBackend):
            def __init__(self, config, seed):
                super().__init__(config, seed)
                self.identity['device_name'] = 'GPU A'
        class OtherGPUCompressor(FakeCompressor):
            def __init__(self, config, seed):
                super().__init__(config, seed)
                self.identity['device_name'] = 'GPU B'
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'between model roles'):
            Experiment(self.config, GPUBackend, OtherGPUCompressor, FakeTokenizer).run()

    def test_frozen_protocol_rejects_altered_data_and_unfrozen_template(self):
        with self.assertRaisesRegex(ValueError, 'Dataset SHA256'):
            Experiment(replace(self.config, expected_dataset_sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'Freeze the template'):
            Experiment(replace(self.config, require_freeze=True))

    def test_changed_source_requires_kernel_restart(self):
        with patch('contractnli.experiment.SOURCE_AT_IMPORT', {}):
            with self.assertRaisesRegex(ValueError, 'Restart the Jupyter kernel'):
                self.run_experiment()

    @unittest.skipUnless(importlib.util.find_spec('llmlingua'), 'optional llmlingua not installed')
    def test_compressor_initialization_failure_releases_loaded_model(self):
        from contractnli.compression import LongLLMLinguaBackend
        with patch('contractnli.compression.TransformersBackend') as model, \
             patch('llmlingua.PromptCompressor.__init__', side_effect=RuntimeError('initialization failed')):
            with self.assertRaisesRegex(RuntimeError, 'initialization failed'):
                LongLLMLinguaBackend(self.config.compressor, 42)
            model.return_value.close.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional analysis dependencies not installed')
    def test_legacy_audit_is_read_only_and_separates_parsing_policies(self):
        from contractnli.audit import audit_predictions, save_audit
        self.run_experiment()
        p = self.root / 'run/predictions.jsonl'
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        row = next(r for r in rows if r['condition'] == 'contextualized' and r['gold'] == 'Entailment')
        row['prediction'] = None
        row['generation']['text'] = '**Entailment**\nExplanation'
        p.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        before = p.read_bytes()
        audit = save_audit(p, None, self.root / 'audit', bootstrap_samples=100)
        self.assertEqual(audit['format_diagnostics']['contextualized']['first_line_correct_n'], 1)
        self.assertAlmostEqual(audit['analyses']['strict']['conditions']['contextualized']['accuracy'], 2/3)
        self.assertEqual(audit['analyses']['first_line_sensitivity']['conditions']['contextualized']['accuracy'], 1)
        self.assertEqual(before, p.read_bytes())
        self.assertEqual(audit, audit_predictions(p, bootstrap_samples=100))
        with self.assertRaisesRegex(ValueError, 'read-only'):
            save_audit(p, None, p.parent, bootstrap_samples=100)
        with self.assertRaisesRegex(ValueError, 'reproduce'):
            audit_predictions(p, self.root / 'run/metrics.json', bootstrap_samples=100)


class ReproducibilityAndBudgetTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'optional inference dependencies not installed')
    def test_model_revisions_freeze_once_and_detect_changed_protocol(self):
        from contractnli.reproducibility import freeze_config
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'frozen.json'
            with patch('transformers.AutoConfig.from_pretrained', return_value=SimpleNamespace(_commit_hash='a'*40)) as get:
                frozen = freeze_config(Config(), target)
                self.assertEqual(get.call_count, 3)
                self.assertTrue(frozen.require_pinned_revisions)
                self.assertEqual(frozen, load_config(target))
                freeze_config(Config(), target)
                self.assertEqual(get.call_count, 3)
                with self.assertRaisesRegex(ValueError, 'differs'):
                    freeze_config(replace(Config(), summary_max_new_tokens=1024), target)
                with self.assertRaisesRegex(ValueError, 'revision differs'):
                    freeze_config(replace(Config(), judge=replace(Config().judge, revision='b'*40)), target)

    def test_reference_cap_counts_all_attempts_and_never_crops(self):
        controls = []
        def produce(control):
            controls.append(control)
            return Generation('x' * (control + 5), 1, control, .5, 'eos')
        result, seconds, attempts = fit_reference_cap(produce, len, 20, 20, 6, integer_control=True)
        self.assertGreater(len(attempts), 1)
        self.assertEqual(seconds, len(attempts) * .5)
        self.assertEqual(result.text, 'x' * (controls[-1] + 5))
        self.assertLessEqual(len(result.text), 20)
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            fit_reference_cap(lambda c: Generation('x'*100, 1, 1, .1, 'eos'), len, 10, 10, 2, integer_control=True)

    def test_main_profile_is_native_and_not_length_fitted(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/rtx5070_v2.toml')
        self.assertEqual(cfg.judge_decoding, 'constrained')
        self.assertEqual(cfg.summary_length_policy, 'fixed')
        self.assertEqual(cfg.summary_max_new_tokens, 2048)
        self.assertEqual(cfg.compressor_budget_mode, 'rate')
        self.assertEqual(cfg.retention_ratio, .55)
        self.assertEqual(cfg.budget_policy, 'target')


if __name__ == '__main__':
    unittest.main()
