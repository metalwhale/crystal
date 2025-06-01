# See:
# - https://colab.research.google.com/github/unslothai/notebooks/blob/159a958/nb/Qwen2.5_(3B)-GRPO.ipynb

import csv
import json
import os
import random
from pathlib import Path

# Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied
import unsloth
from trl import apply_chat_template
from vllm import SamplingParams

from .data import convert_to_conversational
from .model import build


def eval(lora_dir: os.PathLike, val_dataset_dir: os.PathLike):
    model, tokenizer = build()
    model.load_lora(lora_dir)  # See: https://github.com/unslothai/unsloth/issues/2009#issuecomment-2800792586
    # Read prompt template
    PROMPT_FILE_PATH = Path(__file__).parent / "prompts" / "extract.md"
    prompt_template = ""
    with open(PROMPT_FILE_PATH) as prompt_file:
        prompt_template = prompt_file.read()
    # Read news
    news_list: list[tuple[str, str]] = []
    with open(val_dataset_dir / "val.csv") as val_file:
        val_reader = csv.DictReader(val_file)
        for news in val_reader:
            news_list.append((news["origin_text"], news["truth_fields_str"]))
    # Run evaluation
    random.seed()
    while True:
        command = input("Press 'q' to quit: ")
        if command == "q":
            break
        origin_text, truth_fields_str = random.choice(news_list)
        example = {"prompt": convert_to_conversational(origin_text, truth_fields_str, prompt_template)}
        prompt = apply_chat_template(example, tokenizer)["prompt"]
        sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=1024,  # Intentionally set longer than `max_completion_length` during training
        )
        print("======================================================")
        print("------------------ Origin text ------------------")
        print(origin_text)
        # Pretrained model
        completion = model.fast_generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=None,
        )[0].outputs[0].text
        print("------------------ Pretrained completion ------------------")
        print(completion)
        # Lora model
        completion = model.fast_generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=model.load_lora(lora_dir),
        )[0].outputs[0].text
        print("------------------ Lora completion ------------------")
        print(completion)
        print("------------------ Truth fields ------------------")
        print(json.dumps(json.loads(truth_fields_str), ensure_ascii=False, indent=2))
        print()
