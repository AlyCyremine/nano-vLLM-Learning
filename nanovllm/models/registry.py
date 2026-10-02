def create_model(config):
    if config.is_hybrid:
        from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
        return Qwen3_5ForCausalLM(config.hf_text_config, state_capacity=config.max_num_seqs)
    from nanovllm.models.qwen3 import Qwen3ForCausalLM
    return Qwen3ForCausalLM(config.hf_text_config)
