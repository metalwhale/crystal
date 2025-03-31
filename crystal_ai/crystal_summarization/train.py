# See:
# - https://huggingface.co/docs/trl/v0.15.2/en/grpo_trainer
# - https://colab.research.google.com/github/unslothai/notebooks/blob/159a958/nb/Qwen2.5_(3B)-GRPO.ipynb#scrollTo=ptqkXK2D4d6p

import csv
import datetime
import os

import pylcs
from datasets import load_dataset
# Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied
from unsloth import FastLanguageModel
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer


class LogCallback(TrainerCallback):
    _log_file_path: os.PathLike

    def __init__(self, log_file_path: os.PathLike):
        self._log_file_path = log_file_path

    def on_log(self, args, state, control, logs: dict[str, float], **kwargs):
        log_file = None
        if not os.path.isfile(self._log_file_path):
            log_file = open(self._log_file_path, "w")
            log_writer = csv.DictWriter(log_file, fieldnames=logs.keys())
            log_writer.writeheader()
            log_writer.writerow(logs)
        else:
            log_file = open(self._log_file_path, "a")
            log_writer = csv.DictWriter(log_file, fieldnames=logs.keys())
            log_writer.writerow(logs)
        log_file.close()


def train(
    train_dataset_dir: os.PathLike,
    val_dataset_dir: os.PathLike,
    train_dir: os.PathLike,
    lora_rank: int = 64,
) -> os.PathLike:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-0.5B",
        max_seq_length=4096,
        load_in_4bit=True,
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
    run_train_dir = train_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = run_train_dir / "output"
    config = GRPOConfig(
        max_prompt_length=1024,
        max_completion_length=512,
        output_dir=output_dir,
        logging_steps=10,
    )
    dataset = load_dataset("csv", data_files={
        "train": f"{str(train_dataset_dir)}/*.csv",
        "val": f"{str(val_dataset_dir)}/*.csv",
    })
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            _penalize_repetition,
            _penalize_nonexistent_words,
            _penalize_improper_len,
            _reward_word_variety,
        ],
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
        callbacks=[
            LogCallback(run_train_dir / "log.csv"),
        ],
    )
    trainer.train()
    return output_dir

####################
# Reward functions
####################
# Ref: https://huggingface.co/docs/trl/v0.15.2/en/dataset_formats#standard


def _penalize_repetition(
    prompts: list[str],
    completions: list[str],
    **kwargs,
) -> list[float]:
    rewards: list[float] = []
    for prompt, completion in zip(prompts, completions):
        # The longer the common substring (repetition) between the prompt and completion,
        # the larger the negative reward (up to `-1` if the completion is fully included in the prompt).
        rewards.append(-pylcs.lcs_string_length(prompt, completion) / len(completion))
    return rewards


def _penalize_nonexistent_words(
    prompts: list[str],
    completions: list[str],
    **kwargs,
) -> list[float]:
    rewards: list[float] = []
    for prompt, completion in zip(prompts, completions):
        # The more words in the completion that don't exist in the prompt,
        # the greater the negative reward.
        rewards.append(-(len(set(completion) - set(prompt))) / len(completion))
    return rewards


def _penalize_improper_len(
    prompts: list[str],
    completions: list[str],
    **kwargs,
) -> list[float]:
    LOWER_BOUND_RATIO = 0.3
    UPPER_BOUND_RATIO = 0.5
    rewards: list[float] = []
    for prompt, completion in zip(prompts, completions):
        lower_bound = LOWER_BOUND_RATIO * len(prompt)
        upper_bound = UPPER_BOUND_RATIO * len(prompt)
        reward = 0
        if len(completion) <= lower_bound:
            # Penalize short completion: reward is 0 at the lower bound and -1 when the length is zero
            reward = len(completion) / lower_bound - 1
        elif len(completion) >= upper_bound:
            # Penalize long completion: reward is 0 at the upper bound and becomes more negative as the length increases
            reward = 1 - len(completion) / upper_bound
        rewards.append(reward)
    return rewards


def _reward_word_variety(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        # The greater the variety of words generated in the completion, the higher the reward
        rewards.append(len(set(completion)) / len(completion))
    return rewards
