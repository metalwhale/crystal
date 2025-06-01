import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path


SUMMARIZATION_TASK = "summarization"
CHATBOT_TASK = "chatbot"
EXTRACTION_TASK = "extraction"


def main():
    logging.basicConfig()
    mode, task_name, storage_dir = sys.argv[1:4]
    storage_dir = Path(storage_dir)
    task_dataset_dir = storage_dir / "dataset" / task_name
    train_dataset_dir = task_dataset_dir / "train"
    val_dataset_dir = task_dataset_dir / "val"
    test_dataset_dir = task_dataset_dir / "test"
    os.makedirs(train_dataset_dir, exist_ok=True)
    os.makedirs(val_dataset_dir, exist_ok=True)
    os.makedirs(test_dataset_dir, exist_ok=True)
    if mode == "data":
        task_data_dir = storage_dir / "data" / task_name
        os.makedirs(task_data_dir, exist_ok=True)
        if task_name == SUMMARIZATION_TASK:
            from crystal.summarization.data import generate_datasets
            start_date = date.today()
            if len(sys.argv) >= 5:
                start_date = datetime.strptime(sys.argv[4], "%Y-%m-%d").date()
            end_date = start_date + timedelta(days=1)
            if len(sys.argv) >= 6:
                end_date = datetime.strptime(sys.argv[5], "%Y-%m-%d").date()
            generate_datasets(
                task_data_dir,
                train_dataset_dir, val_dataset_dir, test_dataset_dir,
                (start_date, end_date),
            )
        elif task_name == CHATBOT_TASK:
            from crystal.chatbot.data import generate_datasets
            generate_datasets(
                task_data_dir,
                train_dataset_dir, val_dataset_dir,
            )
        elif task_name == EXTRACTION_TASK:
            from crystal.extraction.data import generate_datasets
            start_date = date.today()
            if len(sys.argv) >= 5:
                start_date = datetime.strptime(sys.argv[4], "%Y-%m-%d").date()
            end_date = start_date + timedelta(days=1)
            if len(sys.argv) >= 6:
                end_date = datetime.strptime(sys.argv[5], "%Y-%m-%d").date()
            generate_datasets(
                task_data_dir,
                train_dataset_dir, val_dataset_dir,
                (start_date, end_date),
            )
    elif mode == "train":
        task_train_dir = storage_dir / "train" / task_name
        run_train_dir = task_train_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(run_train_dir, exist_ok=True)
        if task_name == SUMMARIZATION_TASK:
            from crystal.summarization.train import train
            train(train_dataset_dir, val_dataset_dir, run_train_dir)
        elif task_name == CHATBOT_TASK:
            from crystal.chatbot.train import train
            train(train_dataset_dir, val_dataset_dir, run_train_dir)
        elif task_name == EXTRACTION_TASK:
            from crystal.extraction.train import train
            train(train_dataset_dir, val_dataset_dir, run_train_dir)
    elif mode == "eval":
        if task_name == SUMMARIZATION_TASK:
            from crystal.summarization.eval import eval
            eval(sys.argv[4])
        elif task_name == CHATBOT_TASK:
            from crystal.chatbot.eval import eval
            eval(sys.argv[4])
        elif task_name == EXTRACTION_TASK:
            from crystal.extraction.eval import eval
            eval(sys.argv[4], val_dataset_dir)


if __name__ == "__main__":
    main()
