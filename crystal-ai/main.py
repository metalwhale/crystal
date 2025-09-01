import argparse
import datetime
import logging
import os
import pathlib

from dotenv import load_dotenv


def main():
    load_dotenv()
    logging.basicConfig()
    # Config the argument parser
    parser = argparse.ArgumentParser(description="Crystal AI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    subparsers.add_parser("data", help="Generate data")
    subparsers.add_parser("train", help="Train the model")
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument("--lora-dir", required=True, help="Path to the trained LoRA subdirectory")
    args = parser.parse_args()
    # Create dataset directories
    storage_dir = pathlib.Path(__file__).resolve().parent.parent / "storage"
    train_dataset_dir = storage_dir / "dataset" / "train"
    val_dataset_dir = storage_dir / "dataset" / "val"
    os.makedirs(train_dataset_dir, exist_ok=True)
    os.makedirs(val_dataset_dir, exist_ok=True)
    # Parse the arguments
    if args.command == "data":
        from crystal.data import fetch_data, generate_datasets

        data_dir = storage_dir / "data"
        os.makedirs(data_dir, exist_ok=True)
        fetch_data(data_dir)
        generate_datasets(data_dir, train_dataset_dir, val_dataset_dir)
    elif args.command == "train":
        from crystal.train import train

        run_train_dir = storage_dir / "train" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train(train_dataset_dir, val_dataset_dir, run_train_dir)
    elif args.command == "evaluate":
        from crystal.train import evaluate

        evaluate(args.lora_dir, val_dataset_dir)


if __name__ == "__main__":
    main()
