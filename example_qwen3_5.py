import argparse
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B text-only inference")
    parser.add_argument("model", help="Local Qwen3.5-0.8B checkpoint directory")
    parser.add_argument("--prompt", default="用中文简要解释什么是 KV cache。")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False,
    )
    llm = LLM(args.model, tensor_parallel_size=1, enforce_eager=True,
              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=args.gpu_memory_utilization)
    try:
        output = llm.generate([prompt], SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens))
        print(tokenizer.decode(output[0]["token_ids"], skip_special_tokens=True))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
