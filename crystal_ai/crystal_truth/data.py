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
    PROMPT_FILE_PATH = Path(__file__).parent / "prompts" / "extract_fields.md"
    start_date, end_date = date_range
    prompt = ""
    with open(PROMPT_FILE_PATH) as prompt_file:
        prompt = prompt_file.read()
    llamacpp_client = openai.OpenAI(
        base_url=f"{llamacpp_root_endpoint}/v1",
        api_key="no-key",
    )
    prompt = prompt.replace("${YEAR}", str(date.today().year))
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
            truth_writer = csv.DictWriter(truth_file, fieldnames=["article_id", "fields"])
            truth_writer.writeheader()
            for news in origin_reader:
                # Ref: https://github.com/metalwhale/chloria/blob/main/chloria-backend/chloria-api/src/execution/ports/repository.rs
                article_id, text = news["article_id"], news["text"]
                if text == "":
                    continue
                _logger.info(f"Generating truth: cur_date={cur_date}, article_id={article_id}")
                news_prompt = prompt.replace("${CONTENT}", text)
                completion = llamacpp_client.chat.completions.create(
                    model="",
                    messages=[
                        {"role": "system", "content": "You are Crystal, an AI assistant."},
                        {"role": "user", "content": news_prompt},
                    ],
                    # TODO: Choose appropriate values
                    max_tokens=os.environ.get("LLAMACPP_MAX_TOKENS", None),
                )
                choice_content = completion.choices[0].message.content
                truth_writer.writerow({"article_id": article_id, "fields": choice_content})
                truth_file.flush()
