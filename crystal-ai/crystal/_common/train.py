import csv
import os
from collections import defaultdict

from transformers import TrainerCallback


class LogCallback(TrainerCallback):
    _log_file_path: os.PathLike
    _log_field_names: list[str]

    def __init__(self, log_file_path: os.PathLike):
        self._log_file_path = log_file_path

    def on_log(self, args, state, control, logs: dict[str, float], **kwargs):
        log_file = None
        creating_log_file = not os.path.isfile(self._log_file_path)
        with open(self._log_file_path, "a") as log_file:
            if creating_log_file:
                self._log_field_names = logs.keys()
            log_writer = csv.DictWriter(log_file, self._log_field_names)
            if creating_log_file:
                log_writer.writeheader()
            # Filter the key, just in case the last `on_log` event contains logs with keys not seen before
            log_writer.writerow({k: v for k, v in logs.items() if k in self._log_field_names})


def find_redundancy_length(text: str, ignored_chars: list[str] = []) -> int:
    # Strides for each character
    char_strides: dict[str, tuple[int, dict[int, int]]] = {}
    for position, char in enumerate(text):
        if char in ignored_chars:
            continue
        if char not in char_strides:
            char_strides[char] = (position, defaultdict(int))
        else:
            last_position, strides = char_strides[char]
            distance = position - last_position
            # A "stride" is an occurrence of the same character at different positions that are spaced by an equal distance
            strides[distance] += 1
            char_strides[char] = (position, strides)
    # Overall strides for all characters
    overall_strides: dict[int, int] = defaultdict(int)
    for char, (_, strides) in char_strides.items():
        for distance, occurrence in strides.items():
            overall_strides[distance] += occurrence
    # Get the longest length of the stride redundancies
    redundancy_length = max(overall_strides.values(), default=0)
    return redundancy_length
