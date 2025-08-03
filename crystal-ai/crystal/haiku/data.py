import csv
import json
import logging
import os
from datetime import date, timedelta

from .._common.data import download_chloria_origin_data

SENTENCE_MIN_LEN = 50

_ORIGIN_SUBDIR_NAME = "origin"
_logger = logging.getLogger(__name__)


def convert_to_conversational(sentence: str, prompt_template: str) -> list[dict[str, str]]:
    prompt: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    text = prompt_template.replace("${CONTENT}", sentence)
    prompt.append({"role": "user", "content": text})
    return prompt


def generate_datasets(
    task_data_dir: os.PathLike,
    train_dataset_dir: os.PathLike,
    val_dataset_dir: os.PathLike,
    date_range: tuple[date, date],
    max_train_count: int = 10000,
    max_val_count: int = 1000,
):
    origin_data_dir = task_data_dir / _ORIGIN_SUBDIR_NAME
    os.makedirs(origin_data_dir, exist_ok=True)
    download_chloria_origin_data(origin_data_dir, date_range)
    start_date, end_date = date_range
    for day_delta in range((end_date - start_date).days):
        cur_date = start_date + timedelta(days=day_delta)
        origin_file_path = task_data_dir / _ORIGIN_SUBDIR_NAME / f"{cur_date}.csv"
        if not os.path.isfile(origin_file_path):
            _logger.warning(f"News file not found: cur_date={cur_date}")
            continue
        train_file_path = train_dataset_dir / f"{cur_date}.csv"
        val_file_path = val_dataset_dir / f"{cur_date}.csv"
        with open(origin_file_path) as origin_file, \
                open(train_file_path, "w") as train_file, \
                open(val_file_path, "w") as val_file:
            origin_reader = csv.DictReader(origin_file)
            train_writer = csv.DictWriter(train_file, ["sentence"])
            val_writer = csv.DictWriter(val_file, ["sentence"])
            train_writer.writeheader()
            val_writer.writeheader()
            train_count, val_count = 0, 0
            for news in origin_reader:
                # Ref: https://github.com/metalwhale/chloria/blob/main/chloria-backend/chloria-api/src/execution/ports/repository.rs
                article_id, text = news["article_id"], news["text"]
                for sentence in text.split("。"):
                    sentence = sentence.strip()
                    if len(sentence) < SENTENCE_MIN_LEN:
                        continue
                    # It looks like the `article_id` only includes hexadecimal characters (the digits 0 to 9 and the letters a to f)
                    if val_count < max_val_count and article_id[0] in ["e", "f"]:
                        val_writer.writerow({"sentence": sentence})
                        val_count += 1
                    elif train_count < max_train_count:
                        train_writer.writerow({"sentence": sentence})
                        train_count += 1
            _logger.info(
                f"Generated datasets: cur_date={cur_date}, "
                f"train_count={train_count}, val_count={val_count}",
            )
