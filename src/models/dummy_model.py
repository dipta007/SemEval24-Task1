import torch
from torch import nn
import lightning as pl
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score


THRESHOLD = 0.5


class DummyModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name, num_labels=1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, text):
        return self.model(**text).logits

    def training_step(self, batch, batch_idx):
        text, label = batch
        y_hat = self(text)
        loss = self.criterion(y_hat.view(-1), label.view(-1))
        self.get_metrics(label.view(-1), torch.sigmoid(y_hat.view(-1)), "train")
        return loss

    def validation_step(self, batch, batch_idx):
        text, label = batch
        y_hat = self(text)
        loss = self.criterion(y_hat.view(-1), label.view(-1))
        self.get_metrics(label.view(-1), torch.sigmoid(y_hat.view(-1)), "valid")
        return loss

    def predict_step(self, batch, batch_idx):
        text, label = batch
        y_hat = self(text)
        return torch.sigmoid(y_hat.view(-1)), label.view(-1)

    def get_metrics(self, y, y_hat, mode):
        y_hat = y_hat > THRESHOLD
        acc = accuracy_score(y.cpu(), y_hat.cpu())
        if mode in ["train", "valid"]:
            self.log(f"{mode}/acc", acc, prog_bar=True, batch_size=self.config.batch_size)
        return acc

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
