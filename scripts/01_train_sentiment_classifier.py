import logging
import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

import __init__
from src import DEVICE
from src.datamodule.sentiment import SentimentDataModule
from src.dataset.sentiment import SentimentDataset
from src.utils.data_helper import get_class_weights, get_data
from src.wrapper.sentiment import SentimentRobertaWrapper

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="01_config",
)
def main(config: DictConfig):
    splitted_data = get_data()
    class_weights = get_class_weights()

    tokenizer = AutoTokenizer.from_pretrained(config["base"]["pretrained_name"])

    data_module = SentimentDataModule(
        train_dataset=SentimentDataset(
            data=splitted_data["train"],
            dataset_type="train",
            tokenizer=tokenizer,
        ),
        val_dataset=SentimentDataset(
            data=splitted_data["val"],
            dataset_type="val",
            tokenizer=tokenizer,
        ),
        test_dataset=SentimentDataset(
            data=splitted_data["test"],
            dataset_type="test",
            tokenizer=tokenizer,
        ),
        real_batch_size=config["base"]["real_batch_size"],
    )

    wrapper = SentimentRobertaWrapper(
        device=DEVICE,
        real_batch_size=config["base"]["real_batch_size"],
        learning_rate=config["base"]["learning_rate"],
        pretrained_name=config["base"]["pretrained_name"],
        class_weights=class_weights,
    )

    save_path = Path.cwd().parents[0] / config["base"]["save_path"]

    trainer = pl.Trainer(
        max_epochs=config["base"]["max_epoch"],
        enable_checkpointing=True,
        strategy="auto",
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        log_every_n_steps=1,
        logger=TensorBoardLogger(
            save_dir=save_path.parent,
            name="tb_logs",
        ),
        callbacks=[
            callbacks.GradientAccumulationScheduler(
                scheduling={0: config["base"]["batch_size"]}
            ),
        ],
    )

    trainer.fit(
        model=wrapper,
        datamodule=data_module,
    )
    test_result = trainer.test(
        model=wrapper,
        datamodule=data_module,
    )

    wrapper.save_student(save_path=save_path)
    wrapper.save_test_results(save_path=save_path)

    logger.info(test_result)
    logger.info(
        classification_report(
            wrapper.test_preds["trues"],
            wrapper.test_preds["preds"],
            target_names=["neutral", "negative"],
        ),
    )


if __name__ == "__main__":
    main()
