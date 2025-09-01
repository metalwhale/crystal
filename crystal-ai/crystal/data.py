import csv
import glob
import logging
import os
import random

import feedparser
import requests
from bs4 import BeautifulSoup


_logger = logging.getLogger(__name__)


def fetch_data(
    data_dir: os.PathLike,
    sentence_num: int = 1000,
    sentence_min_len: int = 50,
):
    # Check if data file exists
    data_file_path = data_dir / "data.csv"
    if os.path.exists(data_file_path):
        _logger.info("Data has already been fetched.")
        return
    # List all news providers
    YAHOO_NEWS_ROOT_URL = "https://news.yahoo.co.jp"
    html = requests.get(f"{YAHOO_NEWS_ROOT_URL}/rss").text
    soup = BeautifulSoup(markup=html, features="html.parser")
    news_provider_list_elem = soup.find(id="contentsWrap").find_all("ul")[-1]  # ニュース提供社
    # Create and write to data file
    with open(data_file_path, "w") as data_file:
        sentence_count = 0
        data_writer = csv.DictWriter(data_file, fieldnames=["sentence"])
        data_writer.writeheader()
        _logger.info(f"Target number of sentences to fetch: {sentence_num}.")
        # Iterate through all news providers
        for news_provider_elem in news_provider_list_elem.find_all("li"):
            feed_url = news_provider_elem.find("a").get("href")
            feed = feedparser.parse(f"{YAHOO_NEWS_ROOT_URL}{feed_url}")
            # Iterate through all entries, each containing a link to a news article
            # NOTE: We simply fetch all entries at the current time, regardless of when they were published.
            #   This means that if we run the program again, these entries may change, resulting in different news.
            for entry in feed["entries"]:
                # Get text from the news article
                news_html = requests.get(entry["link"]).text
                news_soup = BeautifulSoup(markup=news_html, features="html.parser")
                article_text: str = news_soup.find(id="uamods").find(class_="article_body").text
                # Split the article text into sentences
                sentences = [s for s in article_text.split("。") if len(s) >= sentence_min_len]
                data_writer.writerows([{"sentence": s} for s in sentences])
                data_file.flush()
                sentence_count += len(sentences)
                progress = min(sentence_count / sentence_num, 1) * 100
                _logger.info(f"Fetching data: sentence_count={sentence_count} ({progress:.2f}%).")
                if sentence_count >= sentence_num:
                    return


def generate_datasets(
    data_dir: os.PathLike,
    train_dataset_dir: os.PathLike,
    val_dataset_dir: os.PathLike,
    train_val_ratio: tuple[float, float] = (0.9, 0.1),
):
    total_train_count, total_val_count = 0, 0
    train_file_path = train_dataset_dir / "train.csv"
    val_file_path = val_dataset_dir / "val.csv"
    # Create and write to dataset files
    with open(train_file_path, "w") as train_file, open(val_file_path, "w") as val_file:
        # Iterate through all data files in CSV format
        for data_file_path in glob.glob(os.path.join(data_dir, "*.csv")):
            with open(data_file_path) as data_file:
                data_reader = csv.DictReader(data_file)
                train_writer = csv.DictWriter(train_file, ["sentence"])
                val_writer = csv.DictWriter(val_file, ["sentence"])
                train_writer.writeheader()
                val_writer.writeheader()
                # Read sentences from a data file
                sentences = [r["sentence"] for r in data_reader]
                random.shuffle(sentences)
                train_count = int(len(sentences) * train_val_ratio[0] / sum(train_val_ratio))
                # Write sentences to each dataset
                for sentence in sentences[:train_count]:
                    train_writer.writerow({"sentence": sentence})
                for sentence in sentences[train_count:]:
                    val_writer.writerow({"sentence": sentence})
                total_train_count += train_count
                total_val_count += len(sentences) - train_count
        _logger.info(f"Generated datasets: total_train_count={total_train_count}, total_val_count={total_val_count}")
