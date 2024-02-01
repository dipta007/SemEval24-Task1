from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from sentence_transformers import evaluation
from pathlib import Path
import zipfile
from torch import nn
import argparse


LANGS = [
    # "all",
    # 'amh',
    # 'arq',
    "ary",
    "eng",
    "esp",
    "hau",
    "kin",
    "mar",
    "tel",
]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)



def preprocess_df(df):
    if "text1" not in df.columns:
        df["text1"] = df["Text"].map(lambda x: x.split("\n")[0].strip('"'))
    if "text2" not in df.columns:
        df["text2"] = df["Text"].map(lambda x: x.split("\n")[1].strip('"'))
    df = df[["text1", "text2", 'PairID']]
    return df


def load_data(lang):
    df = pd.read_csv(f"./data/Track A/{lang}/{lang}_dev.csv")
    df = preprocess_df(df)
    return df


def get_evaluator(df):
    sentences1 = df["text1"].tolist()
    sentences2 = df["text2"].tolist()
    scores = df["score"].tolist()
    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    return evaluator


def get_submission(args, smodel_name, df):
    print("Test Data Size:", len(df))
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    text1 = df["text1"].tolist()
    text2 = df["text2"].tolist()

    print("Loading model...")
    smodel = SentenceTransformer(smodel_name)
    smodel = smodel.to(DEVICE)

    emb1 = smodel.encode(text1, convert_to_tensor=True, device=DEVICE, show_progress_bar=True, batch_size=args.batch_size)
    emb2 = smodel.encode(text2, convert_to_tensor=True, device=DEVICE, show_progress_bar=True, batch_size=args.batch_size)
    y_hat = cos(emb1, emb2).tolist()
    df["Pred_Score"] = y_hat

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./sbert_sweep/student",
        help="Checkpoint directory",
    )
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--lang", type=str, default="all", help="Which langs to predict?")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="./submit/sbert_teacher_student")
    args = parser.parse_args()

    print(args)

    smodel = f"{args.checkpoint_dir}/{args.exp_name}"

    if args.lang != "all":
        LANGS = [args.lang]

    for lang in LANGS:
        print("Predicting for", lang)
        df = load_data(lang)
        out = get_submission(args, smodel, df)
        path = f"{args.save_dir}/{args.exp_name}"
        Path(path).mkdir(parents=True, exist_ok=True)
        out.to_csv(f"{path}/pred_{lang}_a.csv", index=False, columns=["PairID", "Pred_Score"])

        zip = zipfile.ZipFile(f"{path}/{lang}.zip", 'w', zipfile.ZIP_DEFLATED)
        zip.write(f"{path}/pred_{lang}_a.csv", arcname=f"pred_{lang}_a.csv")
        zip.close()