import os
import lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TranslatedDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

    def get_dataset(self, split):
        data = []
        df = self.dataset[split]
        for _, row in df.iterrows():
            data.append({"text1": row["text1"], "text2": row["text2"], "score": row["Score"]})
        return data

    def prepare_data(self):
        # download, tokenize, etc...
        # only called on 1 GPU/TPU in distributed
        dataset = pd.read_csv(os.path.join(self.config.data_dir, "train_all.csv"))
        train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=self.config.seed, stratify=dataset['lang'])
        print(len(train_dataset), len(val_dataset))

        self.dataset = {
            "train": train_dataset,
            "val": val_dataset,
        }
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
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0,
        )

    def collate_fn(self, batch):
        text1 = [obj["text1"] for obj in batch]
        text2 = [obj["text2"] for obj in batch]
        score = [obj["score"] for obj in batch]

        text1 = self.tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
        text2 = self.tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
        score = torch.tensor(score, dtype=torch.float32)

        return text1, text2, score


if __name__ == "__main__":
    from base_config import get_config

    config = get_config()
    datamodule = TranslatedDataModule(config)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    for batch in datamodule.train_dataloader():
        text1, text2, score = batch
        print(text1["input_ids"].shape)
        print(text2["input_ids"].shape)
        print(score.shape)
        break
