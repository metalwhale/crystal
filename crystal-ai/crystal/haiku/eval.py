# See:
# - https://colab.research.google.com/github/unslothai/notebooks/blob/159a958/nb/Qwen2.5_(3B)-GRPO.ipynb

import csv
import glob
import json
import os
import random
from pathlib import Path

# Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied
import unsloth
from trl import apply_chat_template
from vllm import SamplingParams

from .data import SENTENCE_MIN_LEN, convert_to_conversational
from .model import build


def eval(lora_dir: os.PathLike, val_dataset_dir: os.PathLike):
    model, tokenizer = build()
    model.load_lora(lora_dir)  # See: https://github.com/unslothai/unsloth/issues/2009#issuecomment-2800792586
    # Read prompt template
    PROMPT_FILE_PATH = Path(__file__).parent / "prompts" / "write_haiku.md"
    prompt_template = ""
    with open(PROMPT_FILE_PATH) as prompt_file:
        prompt_template = prompt_file.read()
    # Read sentences
    sentence_list: list[str] = []
    for val_file_path in glob.glob(os.path.join(val_dataset_dir, "*.csv")):
        with open(val_file_path) as val_file:
            val_reader = csv.DictReader(val_file)
            for row in val_reader:
                sentence = row["sentence"]
                if len(sentence) >= SENTENCE_MIN_LEN:
                    sentence_list.append(row["sentence"])
    # Run evaluation
    random.seed()
    while True:
        command = input("Press 'q' to quit: ")
        if command == "q":
            break
        sentence = random.choice(sentence_list)
        example = {"prompt": convert_to_conversational(sentence, prompt_template)}
        prompt = apply_chat_template(example, tokenizer)["prompt"]
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=2048,
        )
        print("======================================================")
        print("------------------ Sentence ------------------")
        print(sentence)
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
