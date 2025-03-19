import logging
import os
from datetime import date, timedelta
from pathlib import Path

from crystal_truth.data import generate_truths, get_origin_news


def main():
    logging.basicConfig()
    DATA_DIR = Path(__file__).parent.parent / "storage" / "data"
    today = date.today()
    get_origin_news(DATA_DIR, (today, today + timedelta(days=1)))
    generate_truths(DATA_DIR, (today, today + timedelta(days=1)))


if __name__ == "__main__":
    main()
