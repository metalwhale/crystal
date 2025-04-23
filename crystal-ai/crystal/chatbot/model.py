from unsloth import FastLanguageModel


def build(lora_rank: int = 64):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=4096,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.8,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def extract_user_content(prompt: str) -> str:
    # Prompt follows the format of chat template
    # Ref: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/5fee7c4/tokenizer_config.json
    # ```
    # <|im_start|>system
    # ${SYSTEM_CONTENT}<|im_end|>
    # <|im_start|>user
    # ${USER_CONTENT}<|im_end|>
    # <|im_start|>assistant
    # ${ASSISTANT_CONTENT}
    # ```
    return prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
