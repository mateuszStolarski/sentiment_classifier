import re
from typing import Any, Literal

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SentimentDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        dataset_type: Literal["train", "test", "val"],
        tokenizer: AutoTokenizer,
    ) -> None:
        self._data = data
        self.dataset_type = dataset_type
        self._tokenizer = tokenizer

    def _preprocess(self, text: str) -> str:
        """All preprocessing is done in this place due to boxing whole task into single notebook."""
        new_text = []
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return emoji_pattern.sub(r"", " ".join(new_text))

    def __getitem__(self, idx: int) -> Any:
        item = self._data.iloc[idx]
        true_label = item["tag"]
        encoding = self._tokenizer(
            self._preprocess(item["text"]),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        observation = {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": true_label,
            "text": self._preprocess(item["text"]),
        }

        return observation

    def __len__(self) -> int:
        return len(self._data)
