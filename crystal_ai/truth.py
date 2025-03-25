import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from crystal_truth.data import generate_truths, get_origin_news


def main():
    logging.basicConfig()
    data_dir = Path(sys.argv[1])
    today = date.today()
    get_origin_news(data_dir, (today, today + timedelta(days=1)))
    generate_truths(data_dir, (today, today + timedelta(days=1)))


if __name__ == "__main__":
    main()
