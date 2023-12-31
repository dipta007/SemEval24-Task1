import sys
import torch
from models.model import TeluguModel
import os
import lightning as pl
from dataloaders.translated_datamodule import TranslatedDataModule, LANGS
import argparse
from pathlib import Path
import pandas as pd
import zipfile


def test(model_path, exp_name):
    model = TeluguModel.load_from_checkpoint(model_path)

    config = model.config
    config.batch_size = 1

    model.eval()

    datamodule = TranslatedDataModule(config)
    datamodule.prepare_data()
    datamodule.setup("test")

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

        Path(f"submit/{exp_name}").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"./submit/{exp_name}/pred_{lang}_a.csv", index=False, columns=['PairID', 'Pred_Score'])
        # os.system(f"zip -r submit/{exp_name}/{lang}.zip submit/{exp_name}/pred_{lang}_a.csv")
        # shutil.make_archive(f"submit/{exp_name}/{lang}", 'zip', f"submit/{exp_name}/pred_{lang}_a.csv")
        zip = zipfile.ZipFile(f"submit/{exp_name}/{lang}.zip", 'w', zipfile.ZIP_DEFLATED)
        zip.write(f"submit/{exp_name}/pred_{lang}_a.csv", arcname=f"pred_{lang}_a.csv")
        zip.close()


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

    test(model_path, args.exp_name)
