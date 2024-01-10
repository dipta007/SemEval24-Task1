from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sentence_transformers import evaluation
import shutil
from pathlib import Path
import os
import sys
import wandb
import argparse

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

STS_MODELS = [
    # "sentence-transformers/gtr-t5-xxl",
    # "sentence-transformers/sentence-t5-xxl",
    "sentence-transformers/all-mpnet-base-v2",
    # "sentence-transformers/all-mpnet-base-v1",
    "sentence-transformers/all-roberta-large-v1",
    # "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/sentence-t5-xl",
    "sentence-transformers/gtr-t5-xl",
    # "sentence-transformers/gtr-t5-large",
    # "sentence-transformers/sentence-t5-large",
    # "sentence-transformers/all-MiniLM-L12-v2",
    # "sentence-transformers/all-MiniLM-L12-v1",
    # "sentence-transformers/all-MiniLM-L6-v2",
]


def get_config():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--smodel", type=str, required=True)
    argparser.add_argument("--lr_scheduler", type=str, required=True)
    argparser.add_argument("--warmup_steps", type=int, required=True)
    argparser.add_argument("--weight_decay", type=float, required=True)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--batch_size", type=int, default=4)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--validate_every", type=int, default=275)
    argparser.add_argument("--save_dir", type=str, default="./sbert_sweep")
    args = argparser.parse_args()
    print(args)
    return args

def preprocess_df(df):
    if "text1" not in df.columns:
        df["text1"] = df["Text"].map(lambda x: x.split("\n")[0].strip('"'))
    if "text2" not in df.columns:
        df["text2"] = df["Text"].map(lambda x: x.split("\n")[1].strip('"'))
    df["score"] = df["Score"].map(lambda x: float(x))
    df = df[["text1", "text2", "score"]]
    return df


def load_data(args):
    dataset = load_dataset("vkpriya/str-2022")
    dataset = pd.DataFrame(dataset["train"])

    dataset = preprocess_df(dataset)
    train_df, val_df = train_test_split(dataset, test_size=0.2, random_state=args.seed)
    val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=args.seed)

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Test size:", len(test_df))

    return train_df, val_df, test_df


def load_translated_data(args):
    langs = ["ary", "esp", "hau", "kin", "mar", "tel"]
    dfs = []
    for lang in langs:
        if lang in ["amh", "arq", "eng"]:
            continue
        print("Loading", lang)
        lang_df = pd.read_csv(f"./data/Track A/{lang}/train_translation.csv")
        lang_df = preprocess_df(lang_df)
        dfs.append(lang_df)

    eng_df = pd.read_csv("./data/Track A/eng/eng_train.csv")
    eng_df = preprocess_df(eng_df)

    eng_df, val_df = train_test_split(eng_df, test_size=0.2, random_state=args.seed)
    val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=args.seed)

    dfs.append(eng_df)
    train_df = pd.concat(dfs)

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Test size:", len(test_df))

    return train_df, val_df, test_df


def get_evaluator(df):
    sentences1 = df["text1"].tolist()
    sentences2 = df["text2"].tolist()
    scores = df["score"].tolist()
    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    return evaluator


def train(args, train_df, val_df, test_df):
    train_examples = []
    for i, row in train_df.iterrows():
        train_examples.append(InputExample(texts=[row["text1"], row["text2"]], label=row["score"]))

    val_evaluator = get_evaluator(val_df)
    test_evaluator = get_evaluator(test_df)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    # fp = open(f"{args.save_dir}/results.txt", "a")

    model_name = args.smodel
    save_model_name = f"{model_name.split('/')[-1]}_wd_{args.weight_decay}_lrs_{args.lr_scheduler}_warms_{args.warmup_steps}"
    print("Training", save_model_name)

    wandb.login()
    wandb.init(
        entity="gcnssdvae",
        project="sem1",
        tags=["sbert", "sweep"],
        name=save_model_name,
        config={
            "model": model_name,
            "lr_scheduler": args.lr_scheduler,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "validate_every": args.validate_every,
            "save_dir": args.save_dir,
        },
    )
    wandb.define_metric("valid/corr", summary="max")

    def log_metrics_to_wandb(score, epoch, steps):
        wandb.log({"valid/corr": score})
        wandb.log({"valid/epoch": epoch})

    model = SentenceTransformer(model_name, device=DEVICE)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=val_evaluator,
        epochs=args.epochs,
        scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_steps=args.validate_every,
        output_path=f"{args.save_dir}/{save_model_name}",
        save_best_model=True,
        show_progress_bar=True,
        callback=log_metrics_to_wandb,
    )

    print("Testing", save_model_name)
    model = SentenceTransformer(f"{args.save_dir}/{save_model_name}", device=DEVICE)
    out = model.evaluate(test_evaluator)
    print("Score:", out)
    wandb.run.summary["test/corr"] = out
    print("\n")
    # format_str = "{:<70} {:<20} {:<10} {:<10} {:<10.7f}"
    # fp.write(format_str.format(model_name, args.lr_scheduler, args.warmup_steps, args.weight_decay, out) + "\n")


if __name__ == "__main__":
    args = get_config()
    train_df, val_df, test_df = load_data(args)
    # train_df, val_df, test_df = load_translated_data()
    train(args, train_df, val_df, test_df)
