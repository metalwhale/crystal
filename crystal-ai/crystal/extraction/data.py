import csv
import json
import logging
import os
import random
import re
from datetime import date, timedelta
from pathlib import Path

import openai

from .._common.data import chat, download_chloria_origin_data

_ORIGIN_SUBDIR_NAME = "origin"
_TRUTH_SUBDIR_NAME = "truth"
_logger = logging.getLogger(__name__)


def convert_to_conversational(origin_text: str, truth_fields_str: str, prompt_template: str) -> list[dict[str, str]]:
    prompt: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    truth_fields: dict[str, dict[str, str]] = json.loads(truth_fields_str)
    field_list = [f"  - {key}: {info['description']}" for key, info in truth_fields.items()]
    text = prompt_template.replace("${CONTENT}", origin_text).replace("${FIELDS}", "\n".join(field_list))
    prompt.append({"role": "user", "content": text})
    return prompt


def generate_datasets(
    task_data_dir: os.PathLike,
    train_dataset_dir: os.PathLike,
    val_dataset_dir: os.PathLike,
    date_range: tuple[date, date],
    train_ratio: float = 0.9,
):
    origin_data_dir = task_data_dir / _ORIGIN_SUBDIR_NAME
    os.makedirs(origin_data_dir, exist_ok=True)
    download_chloria_origin_data(origin_data_dir, date_range)
    all_news_dict = _generate_truths(task_data_dir, date_range)
    all_news_list = [(t, _sanitize(f, t)) for (t, f) in all_news_dict.values()]
    all_news_list = [(t, json.dumps(f, ensure_ascii=False)) for (t, f) in all_news_list if len(f) > 0]
    random.shuffle(all_news_list)
    # Split into train and val
    train_len = int(train_ratio * len(all_news_list))
    val_len = len(all_news_list) - train_len
    train_news_list = all_news_list[:train_len]
    val_news_list = all_news_list[train_len:train_len + val_len]
    train_file_path = train_dataset_dir / "train.csv"
    val_file_path = val_dataset_dir / "val.csv"
    for file_path, news_list in zip(
        [train_file_path, val_file_path],
        [train_news_list, val_news_list],
    ):
        with open(file_path, "w", encoding="utf8") as dataset_file:
            dataset_writer = csv.DictWriter(dataset_file, ["origin_text", "truth_fields_str"])
            dataset_writer.writeheader()
            for (origin_text, truth_fields_str) in news_list:
                dataset_writer.writerow({
                    "origin_text": origin_text,
                    "truth_fields_str": truth_fields_str,
                })
    _logger.info(f"Generated datasets: train_len={len(train_news_list)}, val_len={len(val_news_list)}")


def _generate_truths(task_data_dir: os.PathLike, date_range: tuple[date, date]) -> dict[str, tuple[str, str]]:
    llamacpp_root_endpoint = os.environ["LLAMACPP_SERVER_ROOT_ENDPOINT"]
    llamacpp_client = openai.OpenAI(
        base_url=f"{llamacpp_root_endpoint}/v1",
        api_key="no-key",
    )
    # Read prompt
    PROMPT_FILE_PATH = Path(__file__).parent / "prompts" / "extract_truth.md"
    prompt = ""
    with open(PROMPT_FILE_PATH) as prompt_file:
        prompt = prompt_file.read()
    prompt = prompt.replace("${YEAR}", str(date.today().year))
    # Generate truths
    news_dict = {}
    os.makedirs(task_data_dir / _TRUTH_SUBDIR_NAME, exist_ok=True)
    start_date, end_date = date_range
    for day_delta in range((end_date - start_date).days):
        cur_date = start_date + timedelta(days=day_delta)
        origin_file_path = task_data_dir / _ORIGIN_SUBDIR_NAME / f"{cur_date}.csv"
        if not os.path.isfile(origin_file_path):
            _logger.warning(f"News file not found: cur_date={cur_date}")
            continue
        # Read previously generated truth data
        truth_file_path = task_data_dir / _TRUTH_SUBDIR_NAME / f"{cur_date}.csv"
        has_pregenerated = False
        pregenerated_news_dict = {}
        if os.path.isfile(truth_file_path):
            has_pregenerated = True
            with open(truth_file_path) as truth_file:
                truth_reader = csv.DictReader(truth_file)
                for truth in truth_reader:
                    pregenerated_news_dict[truth["article_id"]] = truth["fields"]
        # Write new truth data
        with open(origin_file_path) as origin_file, open(truth_file_path, "a") as truth_file:
            origin_reader = csv.DictReader(origin_file)
            truth_writer = csv.DictWriter(truth_file, fieldnames=["article_id", "fields"])
            if not has_pregenerated:
                truth_writer.writeheader()
            for news in origin_reader:
                # Ref: https://github.com/metalwhale/chloria/blob/main/chloria-backend/chloria-api/src/execution/ports/repository.rs
                article_id, text = news["article_id"], news["text"]
                if article_id in pregenerated_news_dict:
                    _logger.info(f"Skip pregenerated truth: cur_date={cur_date}, article_id={article_id}")
                    news_dict[article_id] = (text, pregenerated_news_dict[article_id])
                    continue
                if text == "":
                    continue
                _logger.info(f"Generating truth: cur_date={cur_date}, article_id={article_id}")
                fields = chat(llamacpp_client, prompt.replace("${CONTENT}", text))
                truth_writer.writerow({"article_id": article_id, "fields": fields})
                truth_file.flush()
                news_dict[article_id] = (text, fields)
    return news_dict


def _sanitize(truth_fields_str: str, origin_text: str) -> dict[str, dict[str, str]]:
    try:
        truth_fields = json.loads(truth_fields_str.removeprefix("```json").removesuffix("```"))
    except ValueError as e:
        return {}
    if not isinstance(truth_fields, dict):
        return {}
    BLACKLIST_WORDS = ["field_name", "description", "value"]
    sanitized_truth_fields = {}
    for key, info in truth_fields.items():
        if not (
            isinstance(key, str) and isinstance(info, dict)
            and "description" in info and "value" in info
        ):
            continue
        if (
            "field" in key or key in BLACKLIST_WORDS
            or re.search(r"\d$", key) is not None
        ):
            continue
        description = info["description"]
        value = info["value"]
        if (
            not (isinstance(description, str) and isinstance(value, str))
            or len(set(key).intersection(set(value))) > 0
            or len(set(description).intersection(set(value))) > 0
            or "フィールド" in description or description in BLACKLIST_WORDS
            or str(date.today().year) in value or value in BLACKLIST_WORDS
        ):
            continue
        if value not in origin_text:
            continue
        sanitized_truth_fields[key] = info
    return sanitized_truth_fields
