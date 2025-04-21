# See:
# - https://huggingface.co/docs/trl/v0.15.2/en/grpo_trainer
# - https://colab.research.google.com/github/unslothai/notebooks/blob/159a958/nb/Qwen2.5_(3B)-GRPO.ipynb

import csv
import os
from collections import Counter
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
        def _log(prompts: list[str], completions: list[str], truth_response: list[str]) -> list[float]:
            return self._log(prompts, completions, truth_response)
        return [
            self._penalize_prompt_overlap,
            self._penalize_improper_length,
            self._penalize_character_duplication,
            self._penalize_redundancy,
            self._reward_response_intersection,
            _log,
        ]

    @staticmethod
    def _penalize_prompt_overlap(
        prompts: list[str],
        completions: list[str],
        truth_response: list[str],
        **kwargs,
    ) -> list[float]:
        UPPER_BOUND_RATIO = 0.3
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            if len(completion) == 0:
                rewards.append(-1)
                continue
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
        truth_response: list[str],
        **kwargs,
    ) -> list[float]:
        LOWER_BOUND_RATIO = 0.5
        UPPER_BOUND_RATIO = 1.5
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            if len(completion) == 0:
                rewards.append(-1)
                continue
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
        truth_response: list[str],
        **kwargs,
    ) -> list[float]:
        AMPLIFICATION_FACTOR = 5
        UPPER_BOUND_RATIO = 0.7
        rewards: list[float] = []
        for completion in completions:
            if len(completion) == 0:
                rewards.append(-1)
                continue
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
        truth_response: list[str],
        **kwargs,
    ) -> list[float]:
        AMPLIFICATION_FACTOR = 0.25
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            if len(completion) == 0:
                rewards.append(-1)
                continue
            prompt_redundancy_length = max(find_redundancy_length(prompt, ignored_chars=[" "]), 1)
            completion_redundancy_length = max(find_redundancy_length(completion, ignored_chars=[" "]), 1)
            reward = 0
            if completion_redundancy_length >= prompt_redundancy_length:
                # The longer stride redundancies in the completion compared to the prompt, the greater the negative reward
                reward = 1 - completion_redundancy_length / prompt_redundancy_length
            reward *= AMPLIFICATION_FACTOR
            rewards.append(reward)
        return rewards

    @staticmethod
    def _reward_response_overlap(
        prompts: list[str],
        completions: list[str],
        truth_response: list[str],
        **kwargs,
    ) -> list[float]:
        AMPLIFICATION_FACTOR = 100
        rewards: list[float] = []
        for completion, response in zip(completions, truth_response):
            if len(completion) == 0:
                rewards.append(-1)
                continue
            # "Overlap" is the longest common substring length of the completion and the response
            overlap_length = pylcs.lcs_string_length(completion, response)
            # The longer the overlap, the greater the reward
            reward = min(overlap_length / len(completion), 1) * min(overlap_length / len(response), 1)
            reward *= AMPLIFICATION_FACTOR
            rewards.append(reward)
        return rewards

    @staticmethod
    def _reward_response_intersection(
        prompts: list[str],
        completions: list[str],
        truth_response: list[str],
        **kwargs,
    ) -> list[float]:
        AMPLIFICATION_FACTOR = 4
        rewards: list[float] = []
        for completion, response in zip(completions, truth_response):
            if len(completion) == 0:
                rewards.append(-1)
                continue
            # "Intersection" is the total number of shared characters in the completion and response, including duplicates
            shared_chars = Counter(completion) & Counter(response)  # `&` returns the minimum of corresponding counts
            intersection_length = sum(shared_chars.values())
            # The longer the intersection, the greater the reward
            reward = min(intersection_length / len(completion), 1) * min(intersection_length / len(response), 1)
            reward *= AMPLIFICATION_FACTOR
            rewards.append(reward)
        return rewards

    def _log(
        self,
        prompts: list[str],
        completions: list[str],
        truth_response: list[str],
        **kwargs,
    ) -> list[float]:
        log_file_path = self._completion_log_dir / (datetime.now().strftime("%Y%m%d-%H") + ".csv")
        log_row = {"prompt_completion": prompts[0] + completions[0], "truth_response": truth_response[0]}
        creating_log_file = not os.path.isfile(log_file_path)
        with open(log_file_path, "a") as log_file:
            log_writer = csv.DictWriter(log_file, ["prompt_completion", "truth_response"])
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
        temperature=0.8,
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
