import torch
import math
from torch import nn
import lightning as pl
from transformers import AutoModel
from .encoder import Encoder
from scipy import stats
from dataloaders.translated_datamodule import LANGS


class TranslationModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        encoder = AutoModel.from_pretrained(config.model_name)
        if encoder.config.is_encoder_decoder:
            encoder = encoder.get_encoder()
        self.encoder = Encoder(self.config, encoder)

        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.criterion = nn.MSELoss()

    def forward(self, text1, text2):
        enc1 = self.encoder(text1)
        enc2 = self.encoder(text2)

        dis = self.cos(enc1, enc2)
        return dis

    def training_step(self, batch, batch_idx):
        text1, text2, score = batch
        dis = self(text1, text2)
        loss = self.criterion(dis, score)
        log_dict = self.get_metrics(score.view(-1), dis.view(-1), "train")
        log_dict["train/loss"] = loss
        self.log_dict(log_dict, prog_bar=True, sync_dist=self.config.ddp)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        lang = LANGS[dataloader_idx]
        text1, text2, score = batch
        dis = self(text1, text2)
        loss = self.criterion(dis, score)
        log_dict = self.get_metrics(score.view(-1), dis.view(-1), "valid")
        log_dict["valid/loss"] = loss

        for key, val in log_dict.items():
            self.log(f"{lang}/{key}", val, prog_bar=True, add_dataloader_idx=False, sync_dist=self.config.ddp)
        if lang == 'all':
            self.log("valid/loss", loss, prog_bar=True, add_dataloader_idx=False, sync_dist=self.config.ddp)
            self.log("valid/corr", log_dict["valid/corr"], prog_bar=True, add_dataloader_idx=False, sync_dist=self.config.ddp)
        return loss

    def predict_step(self, batch, batch_idx):
        text1, text2, score = batch
        y_hat = self(text1, text2)
        return y_hat.view(-1)

    def get_metrics(self, y, y_hat, mode):
        score = stats.spearmanr(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())[0]
        if math.isnan(score) or math.isinf(score) or score < -1 or score > 1:
            score = 0.0
        log_dict = {}
        log_dict[f"{mode}/corr"] = score
        return log_dict

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
