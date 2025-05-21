# See:
# - https://colab.research.google.com/github/unslothai/notebooks/blob/159a958/nb/Qwen2.5_(3B)-GRPO.ipynb

import os
from pathlib import Path

# Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied
import unsloth
from trl import apply_chat_template
from vllm import SamplingParams

from .data import convert_to_conversational
from .model import build


def eval(lora_dir: os.PathLike):
    model, tokenizer = build()
    while True:
        user_content = input("Message: ")
        if user_content == "":
            break
        example = {"prompt": convert_to_conversational([user_content])}
        prompt = apply_chat_template(example, tokenizer)["prompt"]
        sampling_params = SamplingParams(
            temperature=0.9,
            max_tokens=1024,  # Intentionally set longer than `max_completion_length` during training
        )
        # Pretrained model
        completion = model.fast_generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=None,
        )[0].outputs[0].text
        print("Pretrained completion ====================")
        print(completion)
        print()
        # Lora model
        model.load_lora(lora_dir)  # See: https://github.com/unslothai/unsloth/issues/2009#issuecomment-2800792586
        completion = model.fast_generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=model.load_lora(lora_dir),
        )[0].outputs[0].text
        print("Lora completion ====================")
        print(completion)
        print()
