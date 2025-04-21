import csv
import json
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path

import openai
import requests

_ORIGIN_SUBDIR_NAME = "origin"
_TRUTH_SUBDIR_NAME = "truth"
_logger = logging.getLogger(__name__)


def generate_datasets(
    task_data_dir: os.PathLike,
    train_dataset_dir: os.PathLike,
    val_dataset_dir: os.PathLike,
    test_dataset_dir: os.PathLike,
    date_range: tuple[date, date],
):
    _download_origin_data(task_data_dir, date_range)
    _generate_datasets(task_data_dir, train_dataset_dir, val_dataset_dir, test_dataset_dir, date_range)


def _download_origin_data(task_data_dir: os.PathLike, date_range: tuple[date, date]):
    os.makedirs(task_data_dir / _ORIGIN_SUBDIR_NAME, exist_ok=True)
    CHLORIA_API_ROOT_ENDPOINT = "https://chloria.wave.metalwhale.dev/api"
    start_date, end_date = date_range
    for day_delta in range((end_date - start_date).days):
        cur_date = start_date + timedelta(days=day_delta)
        origin_file_path = task_data_dir / _ORIGIN_SUBDIR_NAME / f"{cur_date}.csv"
        if os.path.isfile(origin_file_path):
            continue
        _logger.info(f"Downloading origin news: cur_date={cur_date}")
        response = requests.post(
            f"{CHLORIA_API_ROOT_ENDPOINT}/authenticate",
            json={
                "api_key": os.environ["CHLORIA_API_KEY"],
                "api_secret": os.environ["CHLORIA_API_SECRET"],
            },
        )
        token = json.loads(response.text)["token"]
        response = requests.get(
            f"{CHLORIA_API_ROOT_ENDPOINT}/news?date={cur_date.isoformat()}",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        with open(origin_file_path, mode="wb") as origin_file:
            origin_file.write(response.content)


def _generate_datasets(
    task_data_dir: os.PathLike,
    train_dataset_dir: os.PathLike,
    val_dataset_dir: os.PathLike,
    test_dataset_dir: os.PathLike,
    date_range: tuple[date, date],
    text_max_len: int = 1600,
):
    start_date, end_date = date_range
    for day_delta in range((end_date - start_date).days):
        cur_date = start_date + timedelta(days=day_delta)
        origin_file_path = task_data_dir / _ORIGIN_SUBDIR_NAME / f"{cur_date}.csv"
        if not os.path.isfile(origin_file_path):
            _logger.warning(f"News file not found: cur_date={cur_date}")
            continue
        train_file_path = train_dataset_dir / f"{cur_date}.csv"
        val_file_path = val_dataset_dir / f"{cur_date}.csv"
        test_file_path = test_dataset_dir / f"{cur_date}.csv"
        with open(Path(__file__).parent / "prompts" / "standard.md") as standard_prompt_file, \
                open(origin_file_path) as origin_file, \
                open(train_file_path, "w") as train_file, \
                open(val_file_path, "w") as val_file, \
                open(test_file_path, "w") as test_file:
            standard_prompt = standard_prompt_file.read()
            origin_reader = csv.DictReader(origin_file)
            train_writer = csv.DictWriter(train_file, ["prompt"])
            val_writer = csv.DictWriter(val_file, ["prompt"])
            test_writer = csv.DictWriter(test_file, ["prompt"])
            train_writer.writeheader()
            val_writer.writeheader()
            test_writer.writeheader()
            train_count, val_count, test_count = 0, 0, 0
            for news in origin_reader:
                # Ref: https://github.com/metalwhale/chloria/blob/main/chloria-backend/chloria-api/src/execution/ports/repository.rs
                article_id, text = news["article_id"], news["text"]
                if text == "" or len(text) >= text_max_len:
                    continue
                prompt = standard_prompt.replace("${CONTENT}", text)
                # It looks like the `article_id` only includes hexadecimal characters (the digits 0 to 9 and the letters a to f)
                if article_id[0] in ["c", "d"]:
                    val_writer.writerow({"prompt": prompt})
                    val_count += 1
                elif article_id[0] in ["e", "f"]:
                    test_writer.writerow({"prompt": prompt})
                    test_count += 1
                else:
                    train_writer.writerow({"prompt": prompt})
                    train_count += 1
            _logger.info(
                f"Generated datasets: cur_date={cur_date}, "
                f"train_count={train_count}, val_count={val_count}, test_count={test_count}",
            )


def _summarize(data_dir: os.PathLike, date_range: tuple[date, date]):
    llamacpp_root_endpoint = os.environ["LLAMACPP_SERVER_ROOT_ENDPOINT"]
    MAX_TRIES = 50
    try_count = 0
    while True:
        try:
            try_count += 1
            # Doc: https://github.com/ggml-org/llama.cpp/blob/b4927/examples/server/README.md#api-endpoints
            response = requests.get(f"{llamacpp_root_endpoint}/health")
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(10)
        if try_count == MAX_TRIES:
            exit(0)
    os.makedirs(data_dir / _TRUTH_SUBDIR_NAME, exist_ok=True)
    start_date, end_date = date_range
    conversational_prompt = ""
    with open(Path(__file__).parent / "prompts" / "conversational.md") as conversational_prompt_file:
        conversational_prompt = conversational_prompt_file.read()
    llamacpp_client = openai.OpenAI(
        base_url=f"{llamacpp_root_endpoint}/v1",
        api_key="no-key",
    )
    for day_delta in range((end_date - start_date).days):
        cur_date = start_date + timedelta(days=day_delta)
        truth_file_path = data_dir / _TRUTH_SUBDIR_NAME / f"{cur_date}.csv"
        if os.path.isfile(truth_file_path):
            continue
        origin_file_path = data_dir / _ORIGIN_SUBDIR_NAME / f"{cur_date}.csv"
        if not os.path.isfile(origin_file_path):
            _logger.warning(f"News file not found: cur_date={cur_date}")
            continue
        with open(origin_file_path) as origin_file, open(truth_file_path, "w") as truth_file:
            origin_reader = csv.DictReader(origin_file)
            truth_writer = csv.DictWriter(truth_file, ["article_id", "summary"])
            truth_writer.writeheader()
            for news in origin_reader:
                # Ref: https://github.com/metalwhale/chloria/blob/main/chloria-backend/chloria-api/src/execution/ports/repository.rs
                article_id, text = news["article_id"], news["text"]
                if text == "":
                    continue
                _logger.info(f"Generating truth: cur_date={cur_date}, article_id={article_id}")
                summary = _chat(llamacpp_client, conversational_prompt.replace("${CONTENT}", text))
                truth_writer.writerow({"article_id": article_id, "summary": summary})
                truth_file.flush()


def _chat(client: openai.OpenAI, prompt: str) -> str:
    choice_content = ""
    if "$" in prompt:
        _logger.warning("Some placeholders haven't been replaced with real values")
    try:
        completion = client.chat.completions.create(
            model="",
            messages=[
                {"role": "system", "content": "You are Crystal, an AI assistant."},
                {"role": "user", "content": prompt},
            ],
            # TODO: Choose appropriate values
            max_tokens=os.environ.get("LLAMACPP_MAX_TOKENS", None),
        )
        choice_content = completion.choices[0].message.content
    except Exception as error:
        _logger.error(f"Unexpected error: error={error}")
    finally:
        return choice_content
