import csv
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path

import openai
import requests

from .._common.data import chat, download_chloria_origin_data

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
    origin_data_dir = task_data_dir / _ORIGIN_SUBDIR_NAME
    os.makedirs(origin_data_dir, exist_ok=True)
    download_chloria_origin_data(origin_data_dir, date_range)
    _generate_truths(task_data_dir, date_range)


def _generate_truths(task_data_dir: os.PathLike, date_range: tuple[date, date]):
    # Wait for the llama.cpp server to wake up
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
        pregenerated_article_ids = []
        if os.path.isfile(truth_file_path):
            has_pregenerated = True
            with open(truth_file_path) as truth_file:
                truth_reader = csv.DictReader(truth_file)
                for truth in truth_reader:
                    pregenerated_article_ids.append(truth["article_id"])
        # Write new truth data
        with open(origin_file_path) as origin_file, open(truth_file_path, "a") as truth_file:
            origin_reader = csv.DictReader(origin_file)
            truth_writer = csv.DictWriter(truth_file, fieldnames=["article_id", "fields"])
            if not has_pregenerated:
                truth_writer.writeheader()
            for news in origin_reader:
                # Ref: https://github.com/metalwhale/chloria/blob/main/chloria-backend/chloria-api/src/execution/ports/repository.rs
                article_id, text = news["article_id"], news["text"]
                if article_id in pregenerated_article_ids:
                    _logger.info(f"Skip pregenerated truth: cur_date={cur_date}, article_id={article_id}")
                    continue
                if text == "":
                    continue
                _logger.info(f"Generating truth: cur_date={cur_date}, article_id={article_id}")
                fields = chat(llamacpp_client, prompt.replace("${CONTENT}", text))
                truth_writer.writerow({"article_id": article_id, "fields": fields})
                truth_file.flush()
