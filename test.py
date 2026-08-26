import os
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
        "Modern large language models require a significant amount of computation during both training and inference. CPUs are designed to handle a relatively small number of complex tasks efficiently, while GPUs contain many smaller processing units that can execute large numbers of operations in parallel. Deep learning workloads involve operations such as matrix multiplication, attention, and other tensor computations that can benefit from this parallelism.\nDuring large language model inference, the system first processes the input prompt and then generates output tokens one at a time. The performance of this process can depend on many factors, including the length of the input prompt, the number of generated tokens, GPU compute capability, memory bandwidth, and how efficiently the inference engine manages requests and memory.\nInference frameworks such as vLLM attempt to improve serving performance by managing GPU resources efficiently and batching requests from multiple users. Understanding how these systems work requires knowledge of both machine learning models and computer systems.\nBased on the information above, explain why GPUs are important for large language model inference. Then briefly explain what kinds of performance bottlenecks an LLM inference system might encounter.",
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
    outputs = llm.generate(prompts, sampling_params)


    for prompt, output in zip(prompts, outputs):
        in_tokens = tokenizer.encode(prompt)
        print(f"\n{in_tokens}")
        print(type(prompt))
        print(f"Prompt: {prompt!r}\n")
        print(f"Input tokens: {len(in_tokens)}\n")

        print(type(output))
        out_tokens = tokenizer.encode(output['text'])
        print(out_tokens)
        print(f"Completion: {output['text']!r}")
        print(f"Output tokens: {len(out_tokens)}")


if __name__ == "__main__":
    main()
