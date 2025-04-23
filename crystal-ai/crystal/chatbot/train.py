# See:
# - https://huggingface.co/docs/trl/v0.15.2/en/grpo_trainer
# - https://colab.research.google.com/github/unslothai/notebooks/blob/159a958/nb/Qwen2.5_(3B)-GRPO.ipynb

import csv
import os
from collections import Counter
from datetime import datetime

from datasets import load_dataset
# Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied
import unsloth
from trl import GRPOConfig, GRPOTrainer, apply_chat_template
from trl.trainer.grpo_trainer import RewardFunc

from .._common.train import LogCallback, find_redundancy_length
from .model import build, extract_user_content


class RewardFunctionBuilder:
    # Doc: https://huggingface.co/docs/trl/v0.15.2/en/dataset_formats#standard

    _completion_log_dir: os.PathLike

    def __init__(self, completion_log_dir: os.PathLike):
        self._completion_log_dir = completion_log_dir

    def build(self) -> list[RewardFunc]:
        def _log(prompts: list[str], completions: list[str], truth_response: list[str]) -> list[float]:
            return self._log(prompts, completions, truth_response)
        return [
            self._score_words,
            self._penalize_improper_length,
            self._penalize_word_duplication,
            self._penalize_redundancy,
            self._penalize_prompt_intersection,
            _log,
        ]

    @staticmethod
    def _score_words(
        prompts: list[str],
        completions: list[str],
        truth_response: list[str],
        **kwargs,
    ) -> list[float]:
        rewards: list[float] = []
        for completion in completions:
            if len(completion) == 0:
                rewards.append(-1)
                continue
            reward = 0
            completion_words = completion.lower().split(" ")
            # Penalize blacklisted words
            for word in ["tôi", "xin lỗi", "bạn", "đúng", "cảm ơn", "chúc", "đừng"]:
                reward += -0.1 * completion_words.count(word)
            for word in [".", ",", "!"]:
                reward += -0.05 * completion_words.count(word)
            # Reward whitelisted words
            for word in ["em"]:
                if word in completion_words:
                    reward += 0.1
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
        UPPER_BOUND_RATIO = 1.0
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            prompt = extract_user_content(prompt)
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
    def _penalize_word_duplication(
        prompts: list[str],
        completions: list[str],
        truth_response: list[str],
        **kwargs,
    ) -> list[float]:
        AMPLIFICATION_FACTOR = 1
        UPPER_BOUND_RATIO = 0.1
        rewards: list[float] = []
        for completion in completions:
            if len(completion) == 0:
                rewards.append(-1)
                continue
            completion_words = completion.split(" ")
            upper_bound = UPPER_BOUND_RATIO * len(completion_words)
            # "Duplication" is the number of extra times words appear in the completion
            duplication_count = len(completion_words) - len(set(completion_words))
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
            prompt = extract_user_content(prompt)
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
    def _penalize_prompt_intersection(
        prompts: list[str],
        completions: list[str],
        truth_response: list[str],
        **kwargs,
    ) -> list[float]:
        AMPLIFICATION_FACTOR = 2
        UPPER_BOUND_RATIO = 0.1
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            prompt = extract_user_content(prompt)
            if len(completion) == 0:
                rewards.append(-1)
                continue
            prompt_word_counter = Counter(prompt.split(" "))
            completion_word_counter = Counter(completion.split(" "))
            upper_bound = UPPER_BOUND_RATIO * sum(prompt_word_counter.values())
            # "Intersection" is the total number of shared words in the prompt and completion, including duplicates
            shared_word_counter = prompt_word_counter & completion_word_counter  # `&` returns the minimum of corresponding counts
            intersection_length = sum(shared_word_counter.values())
            reward = 0
            if intersection_length >= upper_bound:
                # Penalize excessive intersection: reward is 0 at the upper bound and becomes more negative as intersection grows
                reward = 1 - intersection_length / upper_bound
            reward *= AMPLIFICATION_FACTOR
            rewards.append(reward)
        return rewards

    @staticmethod
    def _penalize_response_divergence(
        prompts: list[str],
        completions: list[str],
        truth_response: list[str],
        **kwargs,
    ) -> list[float]:
        AMPLIFICATION_FACTOR = 2
        LOWER_BOUND_RATIO = 0.5
        rewards: list[float] = []
        for completion, response in zip(completions, truth_response):
            if len(completion) == 0:
                rewards.append(-1)
                continue
            completion_word_counter = Counter(completion.split(" "))
            response_word_counter = Counter(response.split(" "))
            lower_bound = LOWER_BOUND_RATIO * sum(response_word_counter.values())
            # "Intersection" is the total number of shared words in the completion and response, including duplicates
            shared_word_counter = completion_word_counter & response_word_counter  # `&` returns the minimum of corresponding counts
            intersection_length = sum(shared_word_counter.values())
            reward = 0
            if intersection_length <= lower_bound:
                # Penalize excessive divergence: reward is 0 at the lower bound and becomes more negative as intersection shrinks
                reward = intersection_length / lower_bound - 1
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
        AMPLIFICATION_FACTOR = 10
        rewards: list[float] = []
        for completion, response in zip(completions, truth_response):
            if len(completion) == 0:
                rewards.append(-1)
                continue
            completion_word_counter = Counter(completion.split(" "))
            response_word_counter = Counter(response.split(" "))
            # "Intersection" is the total number of shared words in the completion and response, including duplicates
            shared_word_counter = completion_word_counter & response_word_counter  # `&` returns the minimum of corresponding counts
            intersection_length = sum(shared_word_counter.values())
            # The longer the intersection, the greater the reward
            reward = min(intersection_length / sum(response_word_counter.values()), 1)
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
        log_row = {
            "prompt": extract_user_content(prompts[0]),
            "completion": completions[0],
            "truth_response": truth_response[0],
        }
        creating_log_file = not os.path.isfile(log_file_path)
        with open(log_file_path, "a") as log_file:
            log_writer = csv.DictWriter(log_file, ["prompt", "completion", "truth_response"])
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
        temperature=0.9,
        logging_steps=10,
    )
    dataset = load_dataset("csv", data_files={
        "train": f"{str(train_dataset_dir)}/*.csv",
        "val": f"{str(val_dataset_dir)}/*.csv",
    })

    def convert_to_conversational(example: dict[str, str]) -> dict:
        return {"prompt": [
            {"role": "system", "content": "You're a gentle friend who replies cheerfully in Vietnamese with lots of emojis."},
            {"role": "user", "content": example["prompt"]},
        ]}
    dataset = dataset.map(convert_to_conversational)
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
