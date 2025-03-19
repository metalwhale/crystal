import json
import logging
import os
import requests
from datetime import date, timedelta


CHLORIA_API_ROOT_ENDPOINT = "https://chloria.wave.metalwhale.dev/api"

logger = logging.getLogger(__name__)


def get_news(data_dir: os.PathLike, date_range: tuple[date, date]):
    start_date, end_date = date_range
    for day_delta in range((end_date - start_date).days):
        cur_date = start_date + timedelta(days=day_delta)
        logger.info(f"cur_date={cur_date}")
        response = requests.post(
            f"{CHLORIA_API_ROOT_ENDPOINT}/authenticate",
            json={
                "api_key": os.environ["CHLORIA_API_KEY"],
                "api_secret": os.environ["CHLORIA_API_SECRET"],
            })
        token = json.loads(response.text)["token"]
        response = requests.get(
            f"{CHLORIA_API_ROOT_ENDPOINT}/news?date={cur_date.isoformat()}",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        articles_file_path = data_dir / f"{cur_date}.csv"
        with open(articles_file_path, mode="wb") as articles_file:
            articles_file.write(response.content)
