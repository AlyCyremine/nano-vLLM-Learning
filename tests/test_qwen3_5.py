import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM, Qwen3_5TextConfig, Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

from nanovllm.engine.state_manager import StateManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.config import Config
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.gated_delta_net import gated_delta_rule
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import reset_context, set_context
from nanovllm.utils.loader import load_model


def tiny_config():
    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, max_position_embeddings=256,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    )
    config._attn_implementation = "eager"
    return config


def forward_requests(model, requests):
    """requests: (stable sequence ID, input token list, previously consumed length)."""
    slots = model.prepare_states([(seq_id, cached) for seq_id, _, cached in requests])
    ids, positions, ranges, cumulative = [], [], [], [0]
    for _, tokens, cached in requests:
        start = len(ids)
        ids.extend(tokens)
        positions.extend(range(cached, cached + len(tokens)))
        ranges.append((start, len(ids)))
        cumulative.append(len(ids))
    device = model.model.embed_tokens.weight.device
    set_context(True, cu_seqlens_q=torch.tensor(cumulative, device=device), request_ranges=ranges, state_indices=slots)
    try:
        hidden = model(torch.tensor(ids, device=device), torch.tensor(positions, device=device))
        return model.lm_head(hidden), model.compute_logits(hidden)
    finally:
        reset_context()


class DeltaRuleTests(unittest.TestCase):
    @torch.inference_mode()
    def test_chunk_and_decode_match_independent_reference(self):
        torch.manual_seed(11)
        for length in (1, 2, 63, 64, 65, 129):
            with self.subTest(length=length):
                q, k, v = [torch.randn(length, 4, 8) for _ in range(3)]
                g, beta = -torch.rand(length, 4), torch.rand(length, 4)
                state = torch.randn(4, 8, 8)
                actual, actual_state = gated_delta_rule(q, k, v, g, beta, state)
                expected, expected_state = torch_recurrent_gated_delta_rule(
                    q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0),
                    initial_state=state.unsqueeze(0), output_final_state=True, use_qk_l2norm_in_kernel=True,
                )
                torch.testing.assert_close(actual, expected[0], atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(actual_state, expected_state[0], atol=2e-5, rtol=2e-5)


class TextModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.reference = HFQwen3_5ForCausalLM(tiny_config()).eval()
        self.model = Qwen3_5ForCausalLM(tiny_config(), state_capacity=2, attention_backend="torch").eval()
        self.model.load_state_dict(self.reference.state_dict(), strict=True)

    def tearDown(self):
        reset_context()

    @torch.inference_mode()
    def test_prefill_and_cached_decode_match_transformers(self):
        tokens = [1, 5, 9, 12, 17, 23, 29]
        ids = torch.tensor([tokens])
        expected = self.reference(ids, use_cache=True)
        actual, last_logits = forward_requests(self.model, [(101, tokens, 0)])
        torch.testing.assert_close(actual, expected.logits[0], atol=3e-5, rtol=3e-4)
        torch.testing.assert_close(last_logits, actual[-1:], atol=1e-7, rtol=2e-6)
        for i, token in enumerate((31, 19, 3)):
            expected = self.reference(torch.tensor([[token]]), past_key_values=expected.past_key_values, use_cache=True)
            actual, _ = forward_requests(self.model, [(101, [token], len(tokens) + i)])
            torch.testing.assert_close(actual, expected.logits[0], atol=3e-5, rtol=3e-4)

    @torch.inference_mode()
    def test_chunked_prefill_matches_full_prefill(self):
        tokens = [1, 8, 13, 21, 34, 2, 3, 5, 11, 17, 19]
        expected, _ = forward_requests(self.model, [(1, tokens, 0)])
        outputs = []
        consumed = 0
        for chunk in (tokens[:2], tokens[2:3], tokens[3:8], tokens[8:]):
            actual, _ = forward_requests(self.model, [(2, chunk, consumed)])
            outputs.append(actual)
            consumed += len(chunk)
        torch.testing.assert_close(torch.cat(outputs), expected, atol=3e-5, rtol=3e-4)

    @torch.inference_mode()
    def test_batch_reordering_and_state_slot_reuse(self):
        a, b = [1, 2, 3], [9, 8, 7, 6, 5]
        expected_a = self.reference(torch.tensor([a]), use_cache=True)
        expected_b = self.reference(torch.tensor([b]), use_cache=True)
        actual, last = forward_requests(self.model, [(100, a, 0), (200, b, 0)])
        torch.testing.assert_close(actual[:len(a)], expected_a.logits[0], atol=3e-5, rtol=3e-4)
        torch.testing.assert_close(actual[len(a):], expected_b.logits[0], atol=3e-5, rtol=3e-4)
        self.assertEqual(tuple(last.shape), (2, 64))
        actual, _ = forward_requests(self.model, [(200, [13], len(b)), (100, [11], len(a))])
        expected_b = self.reference(torch.tensor([[13]]), past_key_values=expected_b.past_key_values)
        expected_a = self.reference(torch.tensor([[11]]), past_key_values=expected_a.past_key_values)
        torch.testing.assert_close(actual, torch.cat((expected_b.logits[0], expected_a.logits[0])), atol=3e-5, rtol=3e-4)
        self.model.state_manager.release([100, 200])
        actual, _ = forward_requests(self.model, [(300, a, 0)])
        expected_a = self.reference(torch.tensor([a]), use_cache=False)
        torch.testing.assert_close(actual, expected_a.logits[0], atol=3e-5, rtol=3e-4)

    @torch.inference_mode()
    def test_weight_namespace_and_missing_weights(self):
        weights = {name.replace("model.", "model.language_model.", 1): tensor.clone()
                   for name, tensor in self.reference.state_dict().items() if name != "lm_head.weight"}
        weights["model.visual.unused.weight"] = torch.zeros(1)
        weights["mtp.unused.weight"] = torch.zeros(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(weights, str(path))
            report = load_model(self.model, directory)
            self.assertEqual(report["skipped_weights"], 2)
            actual, _ = forward_requests(self.model, [(1, [1, 3, 5], 0)])
            expected = self.reference(torch.tensor([[1, 3, 5]]), use_cache=False).logits[0]
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
            del weights["model.language_model.layers.0.linear_attn.dt_bias"]
            save_file(weights, str(path))
            with self.assertRaisesRegex(ValueError, "Missing model weights"):
                load_model(self.model, directory)
            weights["model.language_model.layers.0.linear_attn.dt_bias"] = torch.zeros(1)
            save_file(weights, str(path))
            with self.assertRaisesRegex(ValueError, "Weight shape mismatch"):
                load_model(self.model, directory)

    def test_dtype_conversion_preserves_float32_state_and_rope(self):
        original = self.model.model.layers[-1].self_attn.rotary_emb.inv_freq.clone()
        self.model.to(dtype=torch.bfloat16)
        for layer in self.model.model.layers[:-1]:
            self.assertEqual(layer.linear_attn.recurrent_states.dtype, torch.float32)
            self.assertEqual(layer.linear_attn.conv_states.dtype, torch.bfloat16)
        torch.testing.assert_close(self.model.model.layers[-1].self_attn.rotary_emb.inv_freq, original, atol=0, rtol=0)

    def test_checkpoint_can_store_either_shared_weight_alias(self):
        weights = {name: tensor.clone() for name, tensor in self.reference.state_dict().items()
                   if name != "model.embed_tokens.weight"}
        with tempfile.TemporaryDirectory() as directory:
            save_file(weights, str(Path(directory) / "model.safetensors"))
            load_model(self.model, directory)
        self.assertIs(self.model.lm_head.weight, self.model.model.embed_tokens.weight)
        torch.testing.assert_close(self.model.model.embed_tokens.weight, self.reference.model.embed_tokens.weight)


class StateAndSamplingTests(unittest.TestCase):
    def test_state_capacity_and_preemption(self):
        manager = StateManager(2)
        first = manager.allocate(1, 0)
        second = manager.allocate(2, 0)
        self.assertEqual(manager.allocate(1, 5), first)
        self.assertNotEqual(first, second)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            manager.allocate(3, 0)
        manager.retain([2])
        self.assertEqual(manager.allocate(3, 0), first)
        with self.assertRaisesRegex(RuntimeError, "Missing GDN state"):
            manager.allocate(4, 10)

    def test_sampling_parameters(self):
        self.assertEqual(SamplingParams(temperature=0).temperature, 0)
        for temperature in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                SamplingParams(temperature=temperature)
        with self.assertRaises(ValueError):
            SamplingParams(max_tokens=0)

    def test_mixed_greedy_and_random_sampling(self):
        logits = torch.tensor([[1.0, 4.0, 2.0], [-float("inf"), -float("inf"), 0.0]])
        temperatures = torch.tensor([0.0, 0.7])
        forward = getattr(Sampler.forward, "_torchdynamo_orig_callable", Sampler.forward)
        self.assertEqual(forward(Sampler(), logits, temperatures).tolist(), [1, 2])

    def test_config_normalizes_text_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            Qwen3_5Config(text_config=tiny_config().to_dict()).save_pretrained(directory)
            config = Config(directory)
            self.assertEqual(config.max_num_seqs, 8)
            self.assertTrue(config.enforce_eager)
            self.assertFalse(config.enable_prefix_caching)
            self.assertEqual(config.hf_text_config.hidden_size, 32)
            with self.assertRaisesRegex(ValueError, "tensor_parallel_size"):
                Config(directory, tensor_parallel_size=2)
            with self.assertRaisesRegex(ValueError, "prefix caching"):
                Config(directory, enable_prefix_caching=True)

    def test_scheduler_limits_active_states_and_handles_chunking(self):
        config = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=2, eos=63, eos_token_ids=(62,),
                                 kvcache_block_size=256, num_kvcache_blocks=4, enable_prefix_caching=False)
        scheduler = Scheduler(config)
        first, second = Sequence([1, 2, 3]), Sequence([4, 5, 6])
        scheduler.add(first)
        scheduler.add(second)
        sequences, prefill = scheduler.schedule()
        scheduler.postprocess(sequences, [7], prefill)
        self.assertEqual(first.num_cached_tokens, 2)
        self.assertEqual(first.num_completion_tokens, 0)
        sequences, prefill = scheduler.schedule()
        scheduler.postprocess(sequences, [7], prefill)
        self.assertEqual(first.num_completion_tokens, 1)
        self.assertFalse(second.block_table)
        sequences, prefill = scheduler.schedule()
        self.assertFalse(prefill)
        scheduler.postprocess(sequences, [62], prefill)
        self.assertTrue(first.is_finished)
        sequences, prefill = scheduler.schedule()
        self.assertEqual(sequences, [second])
        self.assertTrue(prefill)
        self.assertFalse(scheduler.block_manager.hash_to_block_id)


if __name__ == "__main__":
    unittest.main()
