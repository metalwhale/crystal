# See:
# - https://huggingface.co/docs/trl/v0.15.2/en/grpo_trainer
# - https://colab.research.google.com/github/unslothai/notebooks/blob/159a958/nb/Qwen2.5_(3B)-GRPO.ipynb

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
# Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied
import unsloth
from trl import GRPOConfig, GRPOTrainer, apply_chat_template
from trl.trainer.grpo_trainer import RewardFunc

from .._common.train import LogCallback
from .data import convert_to_conversational
from .model import build


class RewardFunctionBuilder:
    # Doc: https://huggingface.co/docs/trl/v0.15.2/en/dataset_formats#standard

    _completion_log_dir: os.PathLike

    def __init__(self, completion_log_dir: os.PathLike):
        self._completion_log_dir = completion_log_dir

    def build(self) -> list[RewardFunc]:
        def _log(
            prompts: list[str],
            origin_text: list[str],
            completions: list[str],
            truth_fields_str: list[str],
        ) -> list[float]:
            return self._log(prompts, origin_text, completions, truth_fields_str)
        return [
            self._score_fields,
            _log,
        ]

    @staticmethod
    def _score_fields(
        prompts: list[str],
        origin_text: list[str],
        completions: list[str],
        truth_fields_str: list[str],
        **kwargs,
    ) -> list[float]:
        scores: list[float] = []
        for text, completion, response in zip(origin_text, completions, truth_fields_str):
            truth_fields: dict[str, dict[str, str]] = json.loads(response.removeprefix("```json").removesuffix("```"))
            keys = set()
            if len(completion) == 0:
                scores.append(0)
                continue
            try:
                fields = json.loads(completion.removeprefix("```json").removesuffix("```"))
            except ValueError:
                scores.append(0)
                continue
            if not isinstance(fields, dict):
                scores.append(0)
                continue
            score = 0
            key_score = 1 / len(truth_fields)
            for key in fields.keys():
                if key not in truth_fields:
                    score += key_score * -0.5
                    continue
                if key in keys:
                    score += key_score * -0.5
                    continue
                keys.add(key)
                if not isinstance(fields[key], str):
                    score += key_score * 0.2
                    continue
                value = fields[key]
                if value not in text:
                    score += key_score * 0.4
                    continue
                if value not in truth_fields[key]["value"] and truth_fields[key]["value"] not in value:
                    score += key_score * 0.6
                    continue
                if value != truth_fields[key]["value"]:
                    score += key_score * 0.8
                    continue
                score += key_score
            scores.append(score)
        return scores

    def _log(
        self,
        prompts: list[str],
        origin_text: list[str],
        completions: list[str],
        truth_fields_str: list[str],
        **kwargs,
    ) -> list[float]:
        log_file_path = self._completion_log_dir / (datetime.now().strftime("%Y%m%d-%H") + ".csv")
        log_row = {
            "origin_text": origin_text[0],
            "completion": completions[0],
            "truth_fields_str": truth_fields_str[0],
        }
        creating_log_file = not os.path.isfile(log_file_path)
        with open(log_file_path, "a") as log_file:
            log_writer = csv.DictWriter(log_file, ["origin_text", "completion", "truth_fields_str"])
            if creating_log_file:
                log_writer.writeheader()
            log_writer.writerow(log_row)
        # Dummy rewards
        return [0 for _ in completions]


def train(
    train_dataset_dir: os.PathLike,
    val_dataset_dir: os.PathLike,
    run_train_dir: os.PathLike,
):
    model, tokenizer = build()
    config = GRPOConfig(
        output_dir=run_train_dir / "output",
        num_train_epochs=1.0,
        max_prompt_length=1024,
        max_completion_length=512,
        temperature=0.2,
        logging_steps=10,
    )
    dataset = load_dataset("csv", data_files={
        "train": f"{str(train_dataset_dir)}/*.csv",
        "val": f"{str(val_dataset_dir)}/*.csv",
    })
    PROMPT_FILE_PATH = Path(__file__).parent / "prompts" / "extract.md"
    prompt_template = ""
    with open(PROMPT_FILE_PATH) as prompt_file:
        prompt_template = prompt_file.read()
    dataset = dataset.map(_map_to_conversational, fn_kwargs={"prompt_template": prompt_template})
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})  # Convert back to standard format
    completion_log_dir = run_train_dir / "completion_log"
    os.makedirs(completion_log_dir, exist_ok=True)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=RewardFunctionBuilder(completion_log_dir).build(),
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
        callbacks=[
            LogCallback(run_train_dir / "log.csv"),
        ],
    )
    trainer.train()
    model.save_lora(run_train_dir / "lora")


def _map_to_conversational(example: dict[str, str], prompt_template: str = "") -> dict[str, list[dict[str, str]]]:
    prompt = convert_to_conversational(example["origin_text"], example["truth_fields_str"], prompt_template)
    return {"prompt": prompt}
