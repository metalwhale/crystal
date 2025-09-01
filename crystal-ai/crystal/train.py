import csv
import datetime
import glob
import os
import random
import string
import unicodedata
from enum import Enum, auto
from functools import reduce
from pathlib import Path

import pykakasi
import pylcs
import unsloth  # Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied
from datasets import load_dataset
from nltk.corpus import cmudict
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer, apply_chat_template
from trl.trainer.grpo_trainer import RewardFunc
from unsloth import FastLanguageModel
from vllm import SamplingParams


class Language(Enum):
    EN = auto()
    JP = auto()
    NONE = auto()


class SyllableCounter:
    _kakasi: pykakasi.kakasi
    _cmudict: dict[str, list[list[str]]]

    def __init__(self):
        kakasi = pykakasi.kakasi()
        kakasi.setMode("H", "a")
        kakasi.setMode("K", "a")
        kakasi.setMode("J", "a")
        kakasi.setMode("s", True)
        self._kakasi = kakasi.getConverter()
        self._cmudict = cmudict.dict()

    def count_sentence(self, sentence: str) -> int:
        """
        Count the number of syllables in a sentence. Supports only English and Japanese characters.
        """
        count = 0
        for lang, segment in self._split_into_segments(sentence):
            if lang == Language.EN:
                count += self._count_sentence_en(segment)
            elif lang == Language.JP:
                count += self._count_sentence_jp(segment)
        return count

    def _count_sentence_en(self, sentence: str) -> int:
        """
        Count the number of syllables in a sentence, assuming it is written in English.
        """

        def count_word(x: str):
            return self._count_word_en(self._strip_punctuation(x))

        return reduce(lambda x, y: x + y, map(count_word, sentence.split(" ")))

    def _count_sentence_jp(self, sentence: str) -> int:
        """
        Count the number of syllables in a sentence, assuming it is written in Japanese.
        """
        sentence = self._kakasi.do(sentence)

        def count_word(x: str):
            return self._count_word_jp(self._strip_punctuation(x))

        return reduce(lambda x, y: x + y, map(count_word, sentence.split(" ")))

    @staticmethod
    def _split_into_segments(sentence: str) -> list[tuple[Language, str]]:
        """
        Split a sentence into segments, each in a different language.
        """
        segments = []
        cur_lang = Language.NONE
        cur_segment = ""
        for i in range(len(sentence) + 1):
            lang = Language.NONE
            if i == len(sentence):  # Last character
                lang = Language.NONE
            else:
                lang = SyllableCounter._check_lang(sentence[i])
            if lang != cur_lang and cur_segment != "":
                segments.append((cur_lang, cur_segment))
                cur_segment = ""
            cur_lang = lang
            if lang != Language.NONE:
                cur_segment += sentence[i]
        return segments

    @staticmethod
    def _check_lang(char: str) -> Language:
        if "LATIN" in unicodedata.name(char, ""):
            return Language.EN
        elif (
            "HIRAGANA" in unicodedata.name(char, "")
            or "KATAKANA" in unicodedata.name(char, "")
            or "CJK UNIFIED IDEOGRAPH" in unicodedata.name(char, "")
        ):
            return Language.JP
        else:
            return Language.NONE

    def _count_word_en(self, word: str) -> int:
        """
        Count the number of syllables in an English word.
        """
        if word == "":
            return 0
        try:
            # Try using nltk
            return [len(list(y for y in x if y[-1].isdigit())) for x in self._cmudict[word.lower()]][0]
        except KeyError:
            # If this fails, use the manual method from
            # See: https://stackoverflow.com/questions/46759492/syllable-count-in-python
            word = word.lower()
            count = 0
            vowels = "aeiouy"
            if word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index - 1] not in vowels:
                    count += 1
            if word.endswith("e"):
                count -= 1
            if count == 0:
                count += 1
            return count

    @staticmethod
    def _count_word_jp(word: str) -> int:
        """
        Count the number of syllables in a Japanese word.
        """
        # Combinations of CVC, CV and V are syllables
        if word == "":
            return 0
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
            elif word[index] in vowels and word[index - 1] in vowels and word[index] != word[index - 1]:
                count += 1
        if count == 0:
            count += 1
        return count

    @staticmethod
    def _strip_punctuation(word: str) -> str:
        return "".join([ch for ch in word if ch not in string.punctuation])


class RewardFunctionBuilder:
    _HAIKU_LINE_NUM: int = 3

    _completion_log_dir: os.PathLike
    _syllable_counter: SyllableCounter

    def __init__(self, completion_log_dir: os.PathLike):
        self._completion_log_dir = completion_log_dir
        self._syllable_counter = SyllableCounter()

    def build(self) -> list[RewardFunc]:
        # Doc: https://huggingface.co/docs/trl/v0.20.0/en/dataset_formats#standard
        def _log(
            prompts: list[str],
            sentence: list[str],
            completions: list[str],
            **kwargs,
        ) -> list[float]:
            return self._log(prompts, sentence, completions)

        return [
            self._score_syllable_count,
            self._reward_existent_characters,
            self._penalize_line_overlap,
            _log,
        ]

    def _score_syllable_count(
        self,
        prompts: list[str],
        sentence: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        """
        Reward completions that follow the syllable pattern rule and penalizes those that do not.
        """
        EXTRA_LINE_DISCREPANCY_FLOOR = 1.1  # Greater than 1
        SHORT_LINE_PENALTY_FACTOR = 2.0
        scores: list[float] = []
        for completion in completions:
            lines = completion.split("\n")
            score = 0
            for i, line in enumerate(lines):
                # Haiku has a strict rule: it must contain exactly 3 lines with a syllable pattern of 5-7-5.
                # We extend this rule to lines that exceed 3, so the pattern continues as 5-7-5-7-5-7...
                # This means we allow haikus longer than 3 lines but penalize the extra lines (starting from the 4th).
                actual_syllable_num = self._syllable_counter.count_sentence(line)
                desired_syllable_num = 5 if i % 2 == 0 else 7
                # Calculate the discrepancy between the actual number of syllables and the desired number of syllables.
                discrepancy = actual_syllable_num - desired_syllable_num
                if discrepancy < 0:
                    # Shorter lines incur a greater penalty
                    discrepancy *= SHORT_LINE_PENALTY_FACTOR
                discrepancy = abs(discrepancy) / desired_syllable_num
                if i >= self._HAIKU_LINE_NUM:
                    # For lines beyond the standard number of lines in a haiku,
                    # we set a positive floor for the discrepancy so that we always penalize these extra lines.
                    discrepancy += EXTRA_LINE_DISCREPANCY_FLOOR
                    extra_line_index = i - self._HAIKU_LINE_NUM + 1
                    # The bigger the discrepancy, the higher the penalty.
                    discrepancy *= extra_line_index
                # The smaller the discrepancy, the greater the score.
                score += 1 / self._HAIKU_LINE_NUM * (1 - discrepancy)
            scores.append(score)
        return scores

    @staticmethod
    def _reward_existent_characters(
        prompts: list[str],
        sentence: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        """
        Reward completions that contain characters from the original sentence to help prevent hallucination.
        """
        EXISTENCE_CEIL = 0.8
        scores: list[float] = []
        for sent, completion in zip(sentence, completions):
            # The more characters in the completion that exist in the sentence, the greater the reward.
            sent_characters = set(sent)
            existence_ratio = len([c for c in completion if c in sent_characters]) / len(completion)
            score = min(existence_ratio, EXISTENCE_CEIL)
            scores.append(score)
        return scores

    @classmethod
    def _penalize_line_overlap(
        cls,
        prompts: list[str],
        sentence: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        """
        Penalize completions that repeat the same words across multiple lines.
        """
        scores: list[float] = []
        for completion in completions:
            lines = completion.split("\n")
            score = 0
            # Iterate through all pairs of distinct lines
            for line1, line2 in [
                (l1, l2)
                for i1, l1 in enumerate(lines[: cls._HAIKU_LINE_NUM])
                for i2, l2 in enumerate(lines[: cls._HAIKU_LINE_NUM])
                if i1 < i2
            ]:
                if len(line1) == 0 or len(line2) == 0:
                    continue
                # "Overlap" is the longest common substring
                overlap_length = pylcs.lcs_string_length(line1, line2)
                score -= 1 / cls._HAIKU_LINE_NUM * overlap_length / min(len(line1), len(line2))
            scores.append(score)
        return scores

    def _log(
        self,
        prompts: list[str],
        sentence: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        log_file_path = self._completion_log_dir / (datetime.datetime.now().strftime("%Y%m%d-%H") + ".csv")
        log_row = {
            "sentence": sentence[0],
            "completion": completions[0],
            "syllable_counts": "-".join(
                [str(self._syllable_counter.count_sentence(line)) for line in completions[0].split("\n")]
            ),
        }
        creating_log_file = not os.path.isfile(log_file_path)
        with open(log_file_path, "a") as log_file:
            log_writer = csv.DictWriter(log_file, ["sentence", "completion", "syllable_counts"])
            if creating_log_file:
                log_writer.writeheader()
            log_writer.writerow(log_row)
        # Dummy rewards
        return [0 for _ in completions]


class LogCallback(TrainerCallback):
    _log_file_path: os.PathLike
    _log_field_names: list[str]

    def __init__(self, log_file_path: os.PathLike):
        self._log_file_path = log_file_path

    def on_log(self, args, state, control, logs: dict[str, float], **kwargs):
        creating_log_file = not os.path.isfile(self._log_file_path)
        with open(self._log_file_path, "a") as log_file:
            if creating_log_file:
                self._log_field_names = logs.keys()
            log_writer = csv.DictWriter(log_file, self._log_field_names)
            if creating_log_file:
                log_writer.writeheader()
            # Filter the key, just in case the last `on_log` event contains logs with keys not seen before
            log_writer.writerow({k: v for k, v in logs.items() if k in self._log_field_names})


def train(
    train_dataset_dir: os.PathLike,
    val_dataset_dir: os.PathLike,
    run_train_dir: os.PathLike,
):
    """
    Train a model that takes an arbitrary sentence as input and outputs a haiku, using GRPO and Unsloth.
    See: https://github.com/unslothai/notebooks/blob/edde39f/nb/HuggingFace%20Course-Qwen2.5_(3B)-GRPO.ipynb
    """
    # Build model
    model, tokenizer = _build_model()
    # Load datasets and apply the chat template
    dataset = load_dataset(
        "csv",
        data_files={
            "train": f"{str(train_dataset_dir)}/*.csv",
            "val": f"{str(val_dataset_dir)}/*.csv",
        },
    )
    prompt_template = ""
    with open(Path(__file__).parent / "prompts" / "write_haiku.md") as prompt_file:
        prompt_template = prompt_file.read()
    dataset = dataset.map(_map_to_conversational, fn_kwargs={"prompt_template": prompt_template})
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})  # Convert back to standard format
    # Start training
    completion_log_dir = run_train_dir / "completion_log"
    os.makedirs(completion_log_dir, exist_ok=True)
    config = GRPOConfig(
        output_dir=run_train_dir / "output",
        num_train_epochs=1.0,
        max_prompt_length=1024,
        max_completion_length=256,
        temperature=1.0,  # AFAIU, a high temperature helps the model explore better during training
        logging_steps=10,
        auto_find_batch_size=False,  # See: https://github.com/unslothai/unsloth/issues/3066
    )
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


def evaluate(lora_dir: os.PathLike, val_dataset_dir: os.PathLike):
    model, tokenizer = _build_model()
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
                sentence_list.append(row["sentence"])
    # Run evaluation
    random.seed()
    while True:
        # Read the command
        command = input(
            "Press 'q' to quit, 'r' to randomly pick a sentence from the validation dataset, "
            "or 'i' to input one manually: "
        )
        sentence = ""
        if command == "q":
            break
        elif command == "r":
            sentence = random.choice(sentence_list)
        elif command == "i":
            # We only support generating haikus from Japanese sentences since we currently train using Japanese datasets
            sentence = input("A Japanese sentence: ")
        # Apply the chat template
        example = {"prompt": _convert_to_conversational(sentence, prompt_template)}
        prompt = apply_chat_template(example, tokenizer)["prompt"]
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=2048,
        )
        print("============================================================")
        print("-------------------- Original sentence --------------------")
        print(sentence)
        # Generate a haiku using the pretrained model
        completion = (
            model.fast_generate(
                [prompt],
                sampling_params=sampling_params,
                lora_request=None,
            )[0]
            .outputs[0]
            .text
        )
        print("-------------------- Pretrained completion --------------------")
        print(completion)
        # Generate a haiku using the trained LoRA model
        completion = (
            model.fast_generate(
                [prompt],
                sampling_params=sampling_params,
                lora_request=model.load_lora(lora_dir),
            )[0]
            .outputs[0]
            .text
        )
        print("-------------------- Trained LoRA completion --------------------")
        print(completion)


def _build_model(lora_rank: int = 64):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",  # An SLM is enough
        max_seq_length=2048,
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


def _map_to_conversational(example: dict[str, str], prompt_template: str = "") -> dict[str, list[dict[str, str]]]:
    prompt = _convert_to_conversational(example["sentence"], prompt_template)
    return {"prompt": prompt}


def _convert_to_conversational(sentence: str, prompt_template: str) -> list[dict[str, str]]:
    prompt: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    text = prompt_template.replace("${CONTENT}", sentence)
    prompt.append({"role": "user", "content": text})
    return prompt
