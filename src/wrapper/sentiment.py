import datetime
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy, F1Score, Recall
from transformers import AutoModelForSequenceClassification


class SentimentRobertaWrapper(pl.LightningModule):
    def __init__(
        self,
        device: str,
        real_batch_size: int,
        learning_rate: float,
        pretrained_name: str,
        class_weights: list[float],
    ) -> None:
        super().__init__()
        self._device = device
        self._real_batch_size = real_batch_size
        self._learning_rate = learning_rate
        self._class_weights = class_weights

        self._student = AutoModelForSequenceClassification.from_pretrained(
            pretrained_name
        )
        self._id2label = self._student.config.id2label

        self._metrics = {
            "recall": Recall(task="binary", average="macro").to(device),
            "accuracy": Accuracy(task="binary", average="macro").to(device),
            "f1": F1Score(task="binary", average="macro").to(device),
        }

        self._mapping = {
            "neutral": 0,
            "negative": 1,
        }

        self.test_preds = {
            "trues": [],
            "preds": [],
        }

    def _get_training_labels(self, labels: torch.Tensor) -> torch.Tensor:
        mapping = {0: 1, 1: 2}

        return (
            torch.Tensor([mapping[x.item()] for x in labels])
            .to(self._device)
            .to(torch.int64)
        )

    def process(self, batch: dict[str, torch.Tensor]):
        input_ids = batch.pop("input_ids").to(self._device)
        attention_mask = batch.pop("attention_mask").to(self._device)
        labels = batch.pop("labels").to(self._device).to(torch.int64)

        with torch.no_grad():
            outputs = self._student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=self._get_training_labels(labels),
            )

        predictions = (
            torch.Tensor(
                [
                    self._mapping[self._id2label[prediction[1:].argmax().item() + 1]]
                    for prediction in outputs.logits
                ]
            )
            .to(self._device)
            .to(torch.int64)
        )

        return labels, predictions, outputs.loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self._student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        input_ids = batch.pop("input_ids").to(self._device)
        attention_mask = batch.pop("attention_mask").to(self._device)
        labels = batch.pop("labels").to(self._device).to(torch.int64)
        labels = self._get_training_labels(labels)

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # TODO: remove after test
        # loss_fct = torch. nn.CrossEntropyLoss(weight=torch.tensor([0, 0.5463003264417845, 5.899529964747356]).to(self._device)).to(self._device)
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self._class_weights).to(self._device)
        ).to(self._device)
        loss = loss_fct(
            outputs.logits.view(-1, self._student.num_labels).to(self._device),
            labels.view(-1).to(self._device),
        ).to(self._device)

        self.log(
            name="train_loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self._real_batch_size,
        )
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        labels, predictions, loss = self.process(batch)

        for name, metric in self._metrics.items():
            self.log(
                name=f"val_{name}",
                value=metric(target=labels, preds=predictions),
                on_epoch=True,
                logger=True,
                prog_bar=True,
                batch_size=self._real_batch_size,
            )

        self.log(
            name=f"val_loss",
            value=loss,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=self._real_batch_size,
        )

        return loss

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        labels, predictions, loss = self.process(batch)

        for name, metric in self._metrics.items():
            self.log(
                name=f"test_{name}",
                value=metric(target=labels, preds=predictions),
                on_epoch=True,
                logger=True,
                prog_bar=True,
                batch_size=self._real_batch_size,
            )

        self.test_preds["trues"].extend(labels.tolist())
        self.test_preds["preds"].extend(predictions.tolist())

        # print(labels, predictions, batch['text'])

        self.log(
            name=f"test_loss",
            value=loss,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=self._real_batch_size,
        )

        return loss

    def configure_optimizers(self) -> list:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self._learning_rate,
        )

        return [optimizer]

    def save_test_results(self, save_path: Path) -> None:
        x = datetime.datetime.now()
        x = x.strftime("%y_%m_%d_%H_%M")
        with (save_path / f"{x}_results.json").open("w") as file:
            json.dump(self.test_preds, file)

    def save_student(self, save_path: Path) -> None:
        x = datetime.datetime.now()
        x = x.strftime("%y_%m_%d_%H_%M")
        self._student.save_pretrained(
            save_path / f"{x}_model",
            from_pt=True,
        )

    def load_student(self, model_path: Path) -> None:
        self._student = AutoModelForSequenceClassification.from_pretrained(model_path)
