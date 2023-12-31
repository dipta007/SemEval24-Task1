from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.dropout = nn.Dropout(self.config.enc_dropout)

    def forward(self, text):
        enc = self.encoder(**text)
        enc = enc.last_hidden_state

        if self.config.enc_pooling == "mean":
            enc = self.mean_pooling(enc, text["attention_mask"])
        elif self.config.enc_pooling == "max":
            enc = self.max_pooling(enc, text["attention_mask"])
        elif self.config.enc_pooling == "cls":
            enc = enc[:, 0, :]
        else:
            raise ValueError(f"Invalid pooling: {self.config.enc_pooling}")

        enc = self.dropout(enc)
        return enc


    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def max_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]