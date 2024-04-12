from pathlib import Path

import torch

TAGS_PATH = Path.cwd().parent / "data/dataset_tags.txt"
TEXTS_PATH = Path.cwd().parent / "data/dataset_texts.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 2024
