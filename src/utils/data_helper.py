from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src import RANDOM_STATE, TAGS_PATH, TEXTS_PATH


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        "input_ids": torch.stack([i["input_ids"] for i in batch]),
        "attention_mask": torch.stack([i["attention_mask"] for i in batch]),
        "labels": torch.Tensor([i["labels"] for i in batch]),
        "text": [i["text"] for i in batch],
    }


def get_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    x, x_test, y, y_test = train_test_split(
        df["text"],
        df["tag"],
        test_size=0.1,
        random_state=RANDOM_STATE,
    )
    test_df = pd.DataFrame(data={"text": x_test, "tag": y_test})
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    val_df = pd.DataFrame(data={"text": x_val, "tag": y_val})
    train_df = pd.DataFrame(data={"text": x_train, "tag": y_train})
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def get_data() -> dict[str, pd.DataFrame]:
    tags_df = pd.read_csv(
        TAGS_PATH,
        sep=" ",
        header=None,
        names=["tag"],
    )
    with TEXTS_PATH.open("r", encoding="utf8") as file:
        text_data = file.readlines()
    text_df = pd.DataFrame(data=text_data, columns=["text"])
    df = pd.concat([text_df, tags_df], axis=1)
    return get_splits(df)


def get_class_weights() -> list[float]:
    tags_df = pd.read_csv(
        TAGS_PATH,
        sep=" ",
        header=None,
        names=["tag"],
    )
    computed_class_weights = compute_class_weight(
        class_weight="balanced",
        classes=[0, 1],
        y=tags_df["tag"],
    ).tolist()
    class_weights = [0]
    class_weights.extend(computed_class_weights)
    return class_weights
