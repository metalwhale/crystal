# See:
# - https://huggingface.co/docs/trl/v0.15.2/en/grpo_trainer
# - https://colab.research.google.com/github/unslothai/notebooks/blob/159a958/nb/Qwen2.5_(3B)-GRPO.ipynb

import csv
import os
from datetime import datetime

import pylcs
from datasets import load_dataset
# Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied
import unsloth
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import RewardFunc

from .._common.train import LogCallback, find_redundancy_length
from .model import build


class RewardFunctionBuilder:
    # Doc: https://huggingface.co/docs/trl/v0.15.2/en/dataset_formats#standard

    _completion_log_dir: os.PathLike

    def __init__(self, completion_log_dir: os.PathLike):
        self._completion_log_dir = completion_log_dir

    def build(self) -> list[RewardFunc]:
        def _log(prompts: list[str], completions: list[str]) -> list[float]:
            return self._log(prompts, completions)
        return [
            self._penalize_nonexistent_characters,
            self._penalize_overlap,
            self._penalize_improper_length,
            self._penalize_character_duplication,
            self._penalize_redundancy,
            _log,
        ]

    @staticmethod
    def _penalize_nonexistent_characters(
        prompts: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            # The more characters in the completion that don't exist in the prompt, the greater the negative reward
            reward = -len([c for c in completion if c not in set(prompt)]) / len(completion)
            rewards.append(reward)
        return rewards

    @staticmethod
    def _penalize_overlap(
        prompts: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        UPPER_BOUND_RATIO = 0.1
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            upper_bound = UPPER_BOUND_RATIO * len(completion)
            # "Overlap" is the longest common substring length of the prompt and the completion
            overlap_length = pylcs.lcs_string_length(prompt, completion)
            reward = 0
            if overlap_length >= upper_bound:
                # Penalize long overlap: reward is 0 at the upper bound and becomes more negative as the overlap grows
                reward = 1 - overlap_length / upper_bound
            rewards.append(reward)
        return rewards

    @staticmethod
    def _penalize_improper_length(
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
            completion_length = len(completion)
            reward = 0
            if completion_length <= lower_bound:
                # Penalize short completion: reward is 0 at the lower bound and -1 when the length is zero
                reward = completion_length / lower_bound - 1
            elif completion_length >= upper_bound:
                # Penalize long completion: reward is 0 at the upper bound and becomes more negative as the length grows
                reward = 1 - completion_length / upper_bound
            rewards.append(reward)
        return rewards

    @staticmethod
    def _penalize_character_duplication(
        prompts: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        AMPLIFICATION_FACTOR = 5
        UPPER_BOUND_RATIO = 0.7
        rewards: list[float] = []
        for completion in completions:
            upper_bound = UPPER_BOUND_RATIO * len(completion)
            # "Duplication" is the number of extra times characters appear in the completion
            duplication_count = len(completion) - len(set(completion))
            reward = 0
            if duplication_count >= upper_bound:
                # Penalize excessive duplication: reward is 0 at the upper bound and becomes more negative as duplication grows
                reward = 1 - duplication_count / upper_bound
            reward *= AMPLIFICATION_FACTOR
            rewards.append(reward)
        return rewards

    @staticmethod
    def _penalize_redundancy(
        prompts: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        AMPLIFICATION_FACTOR = 0.25
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            prompt_redundancy_length = max(find_redundancy_length(prompt), 1)
            completion_redundancy_length = max(find_redundancy_length(completion), 1)
            reward = 0
            if completion_redundancy_length >= prompt_redundancy_length:
                # The longer stride redundancies in the completion compared to the prompt, the greater the negative reward
                reward = 1 - completion_redundancy_length / prompt_redundancy_length
            reward *= AMPLIFICATION_FACTOR
            rewards.append(reward)
        return rewards

    @staticmethod
    def _penalize_excessive_line(
        prompts: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        AMPLIFICATION_FACTOR = 0.5
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            prompt_line_count = prompt.count("\n")
            completion_line_count = completion.count("\n")
            reward = 0
            if completion_line_count >= prompt_line_count:
                # The more lines in the completion compared to the prompt, the greater the negative reward
                reward = 1 - completion_line_count / prompt_line_count
            reward *= AMPLIFICATION_FACTOR
            rewards.append(reward)
        return rewards

    @staticmethod
    def _reward_character_variety(
        prompts: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        AMPLIFICATION_FACTOR = 3
        rewards: list[float] = []
        for completion in completions:
            # The greater the variety of characters generated in the completion, the greater the reward
            reward = len(set(completion)) / len(completion)
            reward *= AMPLIFICATION_FACTOR
            rewards.append(reward)
        return rewards

    def _log(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        log_file_path = self._completion_log_dir / (datetime.now().strftime("%Y%m%d-%H") + ".csv")
        log_row = {"prompt_completion": prompts[0] + completions[0]}
        creating_log_file = not os.path.isfile(log_file_path)
        with open(log_file_path, "a") as log_file:
            log_writer = csv.DictWriter(log_file, ["prompt_completion"])
            if creating_log_file:
                log_writer.writeheader()
            log_writer.writerow(log_row)
        # Dummy rewards
        return [0 for c in completions]


def train(
    train_dataset_dir: os.PathLike,
    val_dataset_dir: os.PathLike,
    run_train_dir: os.PathLike,
):
    # TODO: Fix the bug where, starting from a specific step (usually when the learning rate is at its highest),
    # the model begins outputting meaningless completions by repeatedly generating only a few characters.
    # Currently, I'm addressing this issue by designing the reward functions so that the rewards they return
    # fall within ranges that aren't too far apart, using "AMPLIFICATION_FACTOR" variables to control the weights.
    # I'm not sure if this is related to the bug, but it seems to help.
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
