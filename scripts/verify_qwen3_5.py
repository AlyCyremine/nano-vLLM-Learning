"""Compare the actual checkpoint on CUDA, including batching and chunked prefill.

Run from the project root: python -m scripts.verify_qwen3_5 /path/to/checkpoint
"""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from nanovllm import LLM, SamplingParams


@torch.inference_mode()
def reference_generate(model, tokens, max_tokens, eos_token_ids=()):
    inputs = torch.tensor([tokens], device="cuda")
    cache = None
    output = []
    for _ in range(max_tokens):
        result = model(inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
        token = result.logits[0, -1].argmax().item()
        output.append(token)
        cache = result.past_key_values
        if token in eos_token_ids:
            break
        inputs = torch.tensor([[token]], device="cuda")
    return output


def assert_tokens(actual, expected, name):
    if actual != expected:
        position = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
        raise AssertionError(f"{name}: first divergence at output token {position}; native={actual}, reference={expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--report", default=".cache/qwen3_5_validation.json")
    args = parser.parse_args()
    if args.max_tokens < 2:
        parser.error("--max-tokens must be at least 2 to test preemption")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts = [
        "用中文简要解释什么是 KV cache。",
        "What is the difference between a CPU and a GPU? Answer in one sentence.",
        "Compute 17 + 25. Answer briefly.",
        "Read this context and summarize it in one sentence: " + "The inference engine processes prompts and then generates tokens. " * 40,
    ]
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    ) for text in texts]
    eos_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "请只回答一个数字：1+1等于多少？"}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False,
    )
    eos_ids = {tokenizer.eos_token_id}
    reference = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    expected = [reference_generate(reference, prompt, args.max_tokens) for prompt in prompts]
    expected_eos = reference_generate(reference, eos_prompt, 128, eos_ids)
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    context_limit = max(1024, max(map(len, prompts)) + args.max_tokens)
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1, max_num_seqs=2,
              max_num_batched_tokens=128, max_model_len=context_limit, gpu_memory_utilization=0.6)
    try:
        started = perf_counter()
        outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        for i, (output, golden) in enumerate(zip(outputs, expected)):
            assert_tokens(output["token_ids"], golden, f"batch case {i}")
        repeated = llm.generate([prompts[1], prompts[0]], SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True), use_tqdm=False)
        for i, output in zip((1, 0), repeated):
            assert_tokens(output["token_ids"], expected[i], f"reused state case {i}")
        eos_output = llm.generate([eos_prompt], SamplingParams(temperature=0, max_tokens=128), use_tqdm=False)[0]
        assert_tokens(eos_output["token_ids"], expected_eos, "EOS case")
        llm.add_request(prompts[0], SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True))
        for _ in range(min(4, args.max_tokens - 1)):
            llm.step()
        sequence = llm.scheduler.running[0]
        # schedule() removes a victim from the running deque before preempt().
        llm.scheduler.running.remove(sequence)
        llm.scheduler.preempt(sequence)
        while not llm.is_finished():
            llm.step()
        assert_tokens(sequence.completion_token_ids, expected[0], "preempted request")
        if llm.model_runner.model.state_manager.slots:
            raise AssertionError("Finished requests still own GDN state slots")
        report = {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(), "dtype": "bfloat16",
            "prompt_lengths": list(map(len, prompts)), "max_tokens": args.max_tokens,
            "max_num_seqs": 2, "max_num_batched_tokens": 128,
            "greedy_cases_matched": len(outputs) + len(repeated) + 2,
            "preemption_recompute_matched": True,
            "eos_output_length": len(eos_output["token_ids"]),
            "weight_load": llm.model_runner.weight_load_report,
            "kv_layers": llm.model_runner.kv_cache.size(1),
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "samples": [tokenizer.decode(output["token_ids"], skip_special_tokens=True) for output in outputs],
        }
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
