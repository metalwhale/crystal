import csv
import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import requests

_ORIGIN_SUBDIR_NAME = "origin"
_TRUTH_SUBDIR_NAME = "truth"
_logger = logging.getLogger(__name__)


def get_origin_news(data_dir: os.PathLike, date_range: tuple[date, date]):
    os.makedirs(data_dir / _ORIGIN_SUBDIR_NAME, exist_ok=True)
    CHLORIA_API_ROOT_ENDPOINT = "https://chloria.wave.metalwhale.dev/api"
    start_date, end_date = date_range
    for day_delta in range((end_date - start_date).days):
        cur_date = start_date + timedelta(days=day_delta)
        origin_file_path = data_dir / _ORIGIN_SUBDIR_NAME / f"{cur_date}.csv"
        if os.path.isfile(origin_file_path):
            continue
        _logger.info(f"Getting origin news: cur_date={cur_date}")
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


def generate_truths(data_dir: os.PathLike, date_range: tuple[date, date]):
    os.makedirs(data_dir / _TRUTH_SUBDIR_NAME, exist_ok=True)
    PROMPT_FILE_PATH = Path(__file__).parent / "prompts" / "extract_fields.md"
    start_date, end_date = date_range
    prompt = ""
    with open(PROMPT_FILE_PATH) as prompt_file:
        prompt = prompt_file.read()
    prompt = prompt.replace("${YEAR}", str(date.today().year))
    for day_delta in range((end_date - start_date).days):
        cur_date = start_date + timedelta(days=day_delta)
        _logger.info(f"Generating truth: cur_date={cur_date}")
        truth_file_path = data_dir / _TRUTH_SUBDIR_NAME / f"{cur_date}.csv"
        if os.path.isfile(truth_file_path):
            continue
        origin_file_path = data_dir / _ORIGIN_SUBDIR_NAME / f"{cur_date}.csv"
        if not os.path.isfile(origin_file_path):
            _logger.warning(f"News file not found: cur_date={cur_date}")
            continue
        with open(origin_file_path) as origin_file, open(truth_file_path, "w") as truth_file:
            origin_reader = csv.DictReader(origin_file)
            truth_writer = csv.DictWriter(truth_file, fieldnames=["article_id", "fields"])
            truth_writer.writeheader()
            for news in origin_reader:
                # Ref: https://github.com/metalwhale/chloria/blob/main/chloria-backend/chloria-api/src/execution/ports/repository.rs
                article_id, text = news["article_id"], news["text"]
                news_prompt = prompt.replace("${CONTENT}", text)
                truth_writer.writerow({"article_id": article_id, "fields": "{}"})
                truth_file.flush()
