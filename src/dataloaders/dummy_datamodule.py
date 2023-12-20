import os
import lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

    def get_dataset(self, split):
        return self.dataset[split]

    def prepare_data(self):
        # download, tokenize, etc...
        # only called on 1 GPU/TPU in distributed
        self.dataset = load_dataset("imdb")
        train, test = self.dataset["train"], self.dataset["test"]
        train_val = train.train_test_split(test_size=0.1)
        train, val = train_val["train"], train_val["test"]
        test = test.train_test_split(test_size=0.001)["test"]
        self.dataset = {"train": train, "val": val, "test": test}
        pass

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_dataset = self.get_dataset("train")
            self.val_dataset = self.get_dataset("val")
        elif stage == "test":
            self.test_dataset = self.get_dataset("test")
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=1,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=1,
        )

    def collate_fn(self, batch):
        text = [obj["text"] for obj in batch]
        label = [obj["label"] for obj in batch]

        text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        label = torch.tensor(label, dtype=torch.float32)

        return text, label


if __name__ == "__main__":
    from base_config import get_config

    config = get_config()
    datamodule = DummyDataModule(config)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    for batch in datamodule.train_dataloader():
        text, label = batch
        print(text["input_ids"].shape)
        print(label.shape)
        break
