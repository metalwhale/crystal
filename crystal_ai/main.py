import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path


SUMMARIZATION_TASK = "summarization"


def main():
    logging.basicConfig()
    mode, task_name, storage_dir = sys.argv[1:]
    storage_dir = Path(storage_dir)
    train_dataset_dir = storage_dir / "dataset" / "train"
    val_dataset_dir = storage_dir / "dataset" / "val"
    test_dataset_dir = storage_dir / "dataset" / "test"
    os.makedirs(train_dataset_dir, exist_ok=True)
    os.makedirs(val_dataset_dir, exist_ok=True)
    os.makedirs(test_dataset_dir, exist_ok=True)
    output_dir: os.PathLike
    if mode == "data":
        data_dir = storage_dir / "data"
        os.makedirs(data_dir, exist_ok=True)
        if task_name == SUMMARIZATION_TASK:
            from crystal_summarization.data import generate_datasets
            today = date.today()
            generate_datasets(
                data_dir,
                train_dataset_dir, val_dataset_dir, test_dataset_dir,
                (today, today + timedelta(days=1)),
            )
    elif mode == "train":
        train_dir = storage_dir / "train"
        if task_name == SUMMARIZATION_TASK:
            from crystal_summarization.train import train
            task_train_dir = train_dir / SUMMARIZATION_TASK
            os.makedirs(task_train_dir, exist_ok=True)
            output_dir = train(train_dataset_dir, val_dataset_dir, task_train_dir)


if __name__ == "__main__":
    main()
