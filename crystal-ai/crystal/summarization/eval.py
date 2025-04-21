# See:
# - https://colab.research.google.com/github/unslothai/notebooks/blob/159a958/nb/Qwen2.5_(3B)-GRPO.ipynb

import os
from pathlib import Path

# Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied
import unsloth
from vllm import SamplingParams

from .model import build


def eval(lora_dir: os.PathLike):
    model, _ = build()
    prompt = ""
    with open(Path(__file__).parent / "prompts" / "standard.md") as standard_prompt_file:
        standard_prompt = standard_prompt_file.read()
        prompt = standard_prompt.replace("${CONTENT}", "")
    sampling_params = SamplingParams(
        temperature=0.2,
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
