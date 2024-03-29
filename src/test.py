import sys
import torch
from models.model import TranslationModel
import os
import lightning as pl
from dataloaders.translated_datamodule import TranslatedDataModule, LANGS
import argparse
from pprint import pprint
from pathlib import Path
import pandas as pd
from scipy import stats
import zipfile


def test(model_path, exp_name, stage):
    model = TranslationModel.load_from_checkpoint(model_path)

    if stage:
        stage = f"_{stage}"

    
    config = model.config
    print("="*50)
    print("Config:")
    pprint(vars(config))
    print("="*50)

    config.batch_size = 1

    model.eval()

    datamodule = TranslatedDataModule(config)
    datamodule.prepare_data()
    datamodule.setup(f"test{stage}")

    trainer = pl.Trainer()

    for lang in LANGS:
        print(f"Predicting for {lang}...")
        y_hat = trainer.predict(model, datamodule.test_dataloader(lang))
        y_hat = torch.cat(y_hat, dim=0).view(-1)
        y_hat = y_hat.tolist()

        data = datamodule.test_dataset[lang]

        for i in range(len(data)):
            data[i]["score"] = y_hat[i]

        df = pd.DataFrame(data)
        rename_dict = {"text1": "Text1", "text2": "Text2", "score": "Pred_Score", "pair_id": "PairID"}
        df = df.rename(columns=rename_dict)


        if lang == "esp":   # because we dont have spanish gold labels
            Path(f"./data/final_test/final/zipped").mkdir(parents=True, exist_ok=True)
            df.to_csv(f"./data/final_test/final/zipped/{exp_name}{stage}_pred_{lang}_a.csv", index=False, columns=['PairID', 'Pred_Score'])
            zip = zipfile.ZipFile(f"./data/final_test/final/zipped/{exp_name}.zip", 'w', zipfile.ZIP_DEFLATED)
            zip.write(f"./data/final_test/final/zipped/{exp_name}{stage}_pred_{lang}_a.csv", arcname=f"pred_{lang}_a.csv")
            zip.close()
        else:
            df = df[["PairID", "Pred_Score"]]
            gold_df = pd.read_csv(f"./data/Track A/test/{lang}/{lang}_test_with_labels.csv")
            df = df.merge(gold_df, on="PairID")
            df = df[["PairID", "Pred_Score", "Score"]]
            score = stats.spearmanr(df["Pred_Score"], df["Score"])[0]

            print("="*50)
            print(f"Spearman correlation for {lang}: {score:.4f}")
            print("="*50)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/nfs/ada/ferraro/users/sroydip1/semeval24/task1/checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--lang", type=str, default="all", help="Which langs to predict?")
    parser.add_argument("--stage", type=str, default="", help="Stage? ['', 'final']")
    args = parser.parse_args()

    files = os.listdir(f"{args.checkpoint_dir}/{args.exp_name}")

    if len(files) == 0:
        print("No checkpoints found!")
        sys.exit()

    index = 0
    if len(files) > 1:
        print("Multiple checkpoints found!")
        for i, file in enumerate(files):
            print(f"{i}: {file}")
        index = int(input("Enter checkpoint index: "))

    file_name = files[index]

    model_path = f"{args.checkpoint_dir}/{args.exp_name}/{file_name}"

    if args.lang != "all":
        LANGS = [args.lang]
    else:
        LANGS.remove("all")

    print(f"Using {model_path}")
    print(f"Predicting for {LANGS}")
    test(model_path, args.exp_name, args.stage)
