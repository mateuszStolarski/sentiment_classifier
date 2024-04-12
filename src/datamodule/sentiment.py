from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.dataset.sentiment import SentimentDataset
from src.utils.data_helper import collate_fn


class SentimentDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: SentimentDataset | None,
        test_dataset: SentimentDataset | None,
        val_dataset: SentimentDataset | None,
        collate_fn: callable = collate_fn,
        seed: int = 2024,
        num_workers: int = 2,
        real_batch_size: int = 1,
    ) -> None:
        super().__init__()

        self._seed = seed
        self._batch_size = real_batch_size
        self._num_workers = num_workers
        self._collate_fn = collate_fn

        self._train, self._test, self._val = train_dataset, test_dataset, val_dataset

    def setup(
        self,
        stage: str,
    ) -> None:
        assert self._train.dataset_type == "train"
        assert self._test.dataset_type == "test"
        assert self._val.dataset_type == "val"

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._train,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._test,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
