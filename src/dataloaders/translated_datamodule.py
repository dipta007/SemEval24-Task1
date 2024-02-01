import os
import lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LANGS = [
    "all",
    # 'amh', Amharic is not supported
    # 'arq', Algerian Arabic is not supported
    "ary",
    "eng",
    "esp",
    "hau",
    "kin",
    "mar",
    "tel",
]

INFERENCE_SELECTED_SMODEL = "facebook/nllb-200-3.3B"


class TranslatedDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

    def get_dataset(self, split, lang="all"):
        self.dataset = {} 
        self.dataset['train'] = pd.read_csv(os.path.join(self.config.data_dir, "train_all.csv"))
        self.dataset['val'] = pd.read_csv(os.path.join(self.config.data_dir, "val_all.csv"))

        data = []
        df = self.dataset[split]
        if lang != "all":
            df = df[df["lang"] == lang]
        if split != 'train':
            df = df[df["model"] == INFERENCE_SELECTED_SMODEL]
        for _, row in df.iterrows():
            data.append({"text1": row["text1"], "text2": row["text2"], "score": row["Score"]})

        return data

    def prepare_data(self):
        # download, tokenize, etc...
        # only called on 1 GPU/TPU in distri
        pass

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_dataset = self.get_dataset("train")
            self.val_dataset = {}
            for lang in LANGS:
                self.val_dataset[lang] = self.get_dataset("val", lang)
        elif stage == "test":
            self.test_dataset = {}
            for lang in LANGS:
                file_path = os.path.join(self.config.data_dir, f"{lang}/dev_translation.csv")
                if lang == 'all':
                    file_path = os.path.join(self.config.data_dir, f"dev_all.csv")
                df = pd.read_csv(file_path)
                df = df[df["model"] == INFERENCE_SELECTED_SMODEL]
                data = []
                for _, row in df.iterrows():
                    data.append({"text1": row["text1"], "text2": row["text2"], "score": 0, "pair_id": row["PairID"]})
                self.test_dataset[lang] = data
        elif stage == "test_final":
            self.test_dataset = {}
            for lang in LANGS:
                file_path = os.path.join(f"./test_data/Track A/{lang}/test_translation.csv")
                if lang == 'all':
                    file_path = os.path.join("./test_data/Track A/test_all.csv")
                df = pd.read_csv(file_path)
                df = df[df["model"] == INFERENCE_SELECTED_SMODEL]
                data = []
                for _, row in df.iterrows():
                    data.append({"text1": row["text1"], "text2": row["text2"], "score": 0, "pair_id": row["PairID"]})
                self.test_dataset[lang] = data
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
        val_dataloaders = []
        for lang in LANGS:
            val_dataloaders.append(
                DataLoader(
                    self.val_dataset[lang],
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    collate_fn=self.collate_fn,
                    num_workers=0,
                )
            )
        return val_dataloaders

    def test_dataloader(self, lang):
        return DataLoader(
            self.test_dataset[lang],
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
