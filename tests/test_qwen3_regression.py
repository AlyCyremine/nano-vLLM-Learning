"""Optional real-checkpoint regression: set NANOVLLM_QWEN3_MODEL to a local path."""

import json
import os
import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


@unittest.skipUnless(os.environ.get("NANOVLLM_QWEN3_MODEL") and torch.cuda.is_available(), "requires a local Qwen3 checkpoint and CUDA")
class Qwen3RegressionTests(unittest.TestCase):
    @torch.inference_mode()
    def test_greedy_and_prefix_cache_preserve_original_behavior(self):
        path = os.environ["NANOVLLM_QWEN3_MODEL"]
        tokenizer = AutoTokenizer.from_pretrained(path)
        # The original Qwen3 path has BF16 differences from Transformers. This
        # fixture was recorded from the unmodified 516b9dd implementation using argmax.
        fixture = json.loads((Path(__file__).parent / "fixtures/qwen3_0_6b_greedy_baseline.json").read_text())
        prompt, expected = fixture["prompt_ids"], fixture["output_ids"]
        llm = LLM(path, enforce_eager=True, max_model_len=512, max_num_seqs=2, gpu_memory_utilization=0.6)
        try:
            output = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=8, ignore_eos=True), use_tqdm=False)
            self.assertEqual(output[0]["token_ids"], expected)
            long_prompt = tokenizer.encode("Explain the following sentence: " + "The GPU processes many values in parallel. " * 40)
            params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
            first = llm.generate([long_prompt], params, use_tqdm=False)[0]["token_ids"]
            second = llm.generate([long_prompt], params, use_tqdm=False)[0]["token_ids"]
            self.assertEqual(first, second)
            self.assertTrue(llm.scheduler.block_manager.hash_to_block_id)
        finally:
            llm.exit()


if __name__ == "__main__":
    unittest.main()
