import logging
import os
from datetime import date, timedelta
from pathlib import Path

from crystal.data import get_news


def main():
    logging.basicConfig(level=os.environ["CRYSTAL_LOG_LEVEL"].upper())
    DATA_DIR = Path(__file__).parent.parent / "storage" / "data"
    os.makedirs(DATA_DIR, exist_ok=True)
    today = date.today()
    get_news(DATA_DIR, (today, today + timedelta(days=1)))


if __name__ == "__main__":
    main()
