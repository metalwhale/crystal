import csv
import glob
import json
import logging
import os
import random
import re
from collections import defaultdict

TEXT_SPLITTER = "\n"
_logger = logging.getLogger(__name__)


class Message:
    text: str
    ts: str  # timestamp
    user: str

    def __init__(self, text: str, ts: str, user: str) -> "Message":
        self.text = text
        self.ts = ts
        self.user = user


class Conversation:
    messages: list[Message]

    def __init__(self) -> "Conversation":
        self.messages = []


def convert_to_conversational(texts: list[str]) -> list[dict[str, str]]:
    prompt: list[dict[str, str]] = [
        {"role": "system", "content": "You are a sweet and gentle girl with a hint of naughtiness; you always reply concisely in Vietnamese, in a seductive manner."},
    ]
    for i, text in enumerate(texts):
        if i % 2 == 0:
            prompt.append({"role": "user", "content": text})
        else:
            prompt.append({"role": "assistant", "content": text})
    return prompt


def generate_datasets(
    task_data_dir: os.PathLike,
    train_dataset_dir: os.PathLike,
    val_dataset_dir: os.PathLike,
    train_len: int = 6000,
    val_len: int = 400,
    conversation_time_gap: int = 60 * 60,  # In seconds
    text_min_len: int = 128,
    text_max_len: int = 512,
) -> int:
    origin_data_dir = task_data_dir / "origin" / "slack"
    # Divide the chat history into conversations
    conversations: list[Conversation] = []
    for channel in next(os.walk(origin_data_dir))[1]:  # List all subdirectories
        thread_dict = defaultdict(list)
        for json_file_path in sorted(glob.glob(os.path.join(origin_data_dir, channel, "*.json"))):
            with open(json_file_path) as json_file:
                for message_obj in json.load(json_file):
                    if "thread_ts" in message_obj:  # Messages in a thread
                        if message_obj["thread_ts"] == message_obj["ts"]:  # Root message of the thread
                            thread_dict["main"].append(message_obj)
                        thread_dict[message_obj["thread_ts"]].append(message_obj)
                    # Normal messages. See: https://api.slack.com/events/message#subtypes.
                    elif "subtype" not in message_obj:
                        thread_dict["main"].append(message_obj)
        for thread, message_obj_list in thread_dict.items():
            conversation = Conversation()
            for message_obj in message_obj_list:
                if "user" not in message_obj or len(message_obj["text"]) == 0:
                    continue
                text = _preprocess(message_obj["text"])
                last_message = None if len(conversation.messages) == 0 else conversation.messages[-1]
                if last_message is not None:
                    assert (float(message_obj["ts"]) >= float(last_message.ts))
                    if (
                        thread == "main"
                        and float(message_obj["ts"]) - float(last_message.ts) >= conversation_time_gap
                    ):
                        # Create a new conversation, if enough time has passed since the previous message was sent
                        # or the max number of messages has been reached
                        conversations.append(conversation)
                        conversation = Conversation()
                    elif message_obj["user"] == last_message.user:
                        last_message.text += " " + text
                        continue
                conversation.messages.append(Message(text, message_obj["ts"], message_obj["user"]))
            conversations.append(conversation)
    # Convert to strings
    all_text_sequences = [
        [m.text for m in c.messages]
        for c in conversations
        if len(c.messages) >= 2
    ]
    all_text_sequences = [
        s for s in all_text_sequences
        if text_min_len <= sum([len(t) for t in s]) < text_max_len
    ]
    random.shuffle(all_text_sequences)
    # Split into train and val
    train_text_sequences = all_text_sequences[:train_len]
    val_text_sequences = all_text_sequences[train_len:train_len + val_len]
    train_file_path = train_dataset_dir / "train.csv"
    val_file_path = val_dataset_dir / "val.csv"
    for file_path, text_sequences in zip(
        [train_file_path, val_file_path],
        [train_text_sequences, val_text_sequences],
    ):
        with open(file_path, "w", encoding="utf8") as dataset_file:
            dataset_writer = csv.DictWriter(dataset_file, ["raw_prompt", "truth_response"])
            dataset_writer.writeheader()
            for text_sequence in text_sequences:
                if len(text_sequence) % 2 != 0:
                    # The number of texts in a sequence should be even
                    # so that the last text can be assigned to the assistant's `truth_response`.
                    text_sequence.pop()
                dataset_writer.writerow({
                    "raw_prompt": TEXT_SPLITTER.join(text_sequence[:-1]),
                    "truth_response": text_sequence[-1],
                })
    _logger.info(f"Generated datasets: train_len={len(train_text_sequences)}, val_len={len(val_text_sequences)}")
    return len(all_text_sequences)


def _preprocess(text: str) -> str:
    text = text.rstrip()
    text = re.sub(r"(\n)+", ", ", text)
    text += "."
    return text
