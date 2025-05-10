import json
import logging
import os
from datetime import date, timedelta

import openai
import requests

_logger = logging.getLogger(__name__)


def download_chloria_origin_data(origin_data_dir: os.PathLike, date_range: tuple[date, date]):
    CHLORIA_API_ROOT_ENDPOINT = "https://chloria.wave.metalwhale.dev/api"
    start_date, end_date = date_range
    for day_delta in range((end_date - start_date).days):
        cur_date = start_date + timedelta(days=day_delta)
        origin_file_path = origin_data_dir / f"{cur_date}.csv"
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


def chat(client: openai.OpenAI, prompt: str) -> str:
    choice_content = ""
    if "${" in prompt:
        _logger.warning("Some placeholders haven't been replaced with real values")
    try:
        max_tokens = os.environ.get("LLAMACPP_MAX_TOKENS", None)
        if max_tokens is not None:
            try:
                max_tokens = int(max_tokens)
            except ValueError:
                max_tokens = None
        completion = client.chat.completions.create(
            model="",
            messages=[
                {"role": "system", "content": "You are Crystal, an AI assistant."},
                {"role": "user", "content": prompt},
            ],
            # TODO: Choose appropriate values
            max_tokens=max_tokens,
        )
        choice_content = completion.choices[0].message.content
    except Exception as error:
        _logger.error(f"Unexpected error: error={error}")
    finally:
        return choice_content
