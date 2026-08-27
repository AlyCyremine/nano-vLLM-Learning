import os, time
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "What is a GPU?",
        "Explain the difference between a CPU and a GPU. Why are GPUs generally more suitable for deep learning workloads? Please explain it in simple terms and give one concrete example.",
        # "Modern large language models require a significant amount of computation during both training and inference. CPUs are designed to handle a relatively small number of complex tasks efficiently, while GPUs contain many smaller processing units that can execute large numbers of operations in parallel. Deep learning workloads involve operations such as matrix multiplication, attention, and other tensor computations that can benefit from this parallelism.\nDuring large language model inference, the system first processes the input prompt and then generates output tokens one at a time. The performance of this process can depend on many factors, including the length of the input prompt, the number of generated tokens, GPU compute capability, memory bandwidth, and how efficiently the inference engine manages requests and memory.\nInference frameworks such as vLLM attempt to improve serving performance by managing GPU resources efficiently and batching requests from multiple users. Understanding how these systems work requires knowledge of both machine learning models and computer systems.\nBased on the information above, explain why GPUs are important for large language model inference. Then briefly explain what kinds of performance bottlenecks an LLM inference system might encounter.",
    ]

    for prompt in prompts:
        input_tokens = tokenizer.encode(prompt)
        print(input_tokens)
        print(len(input_tokens))

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # outputs = llm.generate(prompts, sampling_params)


    for prompt in prompts:
        start = time.perf_counter()
        output = llm.generate([prompt], sampling_params)
        latency = time.perf_counter() - start
        print(f"\nLatency:{latency}")

        print(type(prompt))
        in_tokens_ids = tokenizer.encode(prompt)
        print(f"Input Tokens IDs: {in_tokens_ids}")
        print(f"Prompt: {prompt!r}")
        print(f"Input tokens: {len(in_tokens_ids)}\n")

        # print("output type:", type(output))
        # print("output:", output)
        # print("output[0] type:", type(output[0]))
        # print("output[0]:", output[0])

        print(type(output))
        out_tokens_ids = output[0]['token_ids']
        print(f"Completion: {output[0]['text']!r}")        
        print(f"Output Tokens IDs: {out_tokens_ids}")
        print(f"Output tokens: {len(out_tokens_ids)}\n")

        # TODOs:
        # 为什么 output_tokens / total_latency = 33 tok/s, 但是 Decode = 74 tok/s
        # 得去看看 Nano-vLLM 怎么算的 Prefill/Decode throughput.


if __name__ == "__main__":
    main()
