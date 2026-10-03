"""Failure-path and admission regressions; no model weights or GPU required."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def scheduler_config():
    return SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=256, eos=63,
                           eos_token_ids=(63,), kvcache_block_size=256,
                           num_kvcache_blocks=1, enable_prefix_caching=False)


class EngineBoundaryTests(unittest.TestCase):
    def test_request_must_fit_physical_cache_even_without_other_requests(self):
        scheduler = Scheduler(scheduler_config())
        for prompt_length, max_tokens in ((257, 1), (256, 2)):
            with self.subTest(prompt_length=prompt_length, max_tokens=max_tokens):
                with self.assertRaisesRegex(ValueError, "KV cache capacity"):
                    scheduler.add(Sequence([1] * prompt_length, SamplingParams(max_tokens=max_tokens)))
                self.assertTrue(scheduler.is_finished())
        # The final sampled token needs no KV entry: 256 prompt + 1 output fits.
        scheduler.add(Sequence([1] * 256, SamplingParams(max_tokens=1)))
        sequences, prefill = scheduler.schedule()
        scheduler.postprocess(sequences, [2], prefill)
        self.assertTrue(scheduler.is_finished())

    def test_failed_initialization_restores_torch_defaults_and_process_group(self):
        config = SimpleNamespace(kvcache_block_size=256, enforce_eager=True,
                                 tensor_parallel_size=1, model_dtype=torch.bfloat16, model="unused")
        dtype, device = torch.get_default_dtype(), torch.get_default_device()
        for failure_stage in ("create_model", "load_model"):
            with self.subTest(failure_stage=failure_stage):
                with patch("nanovllm.engine.model_runner.dist.init_process_group"), \
                     patch("nanovllm.engine.model_runner.dist.destroy_process_group") as destroy, \
                     patch("nanovllm.engine.model_runner.torch.cuda.set_device"), \
                     patch("nanovllm.engine.model_runner.create_model"), \
                     patch(f"nanovllm.engine.model_runner.{failure_stage}", side_effect=ValueError("bad checkpoint")):
                    try:
                        with self.assertRaisesRegex(ValueError, "bad checkpoint"):
                            ModelRunner(config, 0, [])
                        self.assertEqual(torch.get_default_dtype(), dtype)
                        self.assertEqual(torch.get_default_device(), device)
                        destroy.assert_called_once_with()
                    finally:
                        torch.set_default_dtype(dtype)
                        torch.set_default_device(device)

    def test_rejected_generate_batch_leaves_no_partial_requests(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine.config = SimpleNamespace(max_model_len=512, hf_text_config=SimpleNamespace(vocab_size=64))
        engine.scheduler = Scheduler(scheduler_config())
        params = SamplingParams(max_tokens=1)
        for prompts, sampling_params in (([[1], []], params), ([[1], [64]], params),
                                          ([[1], [1] * 257], params), ([[1], [2]], [params])):
            with self.subTest(lengths=list(map(len, prompts))):
                with self.assertRaises(ValueError):
                    engine.generate(prompts, sampling_params, use_tqdm=False)
                self.assertTrue(engine.scheduler.is_finished())


if __name__ == "__main__":
    unittest.main()
