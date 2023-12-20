import sys
import torch
from models.dummy_model import DummyModel
import os
import lightning as pl
from dataloaders.dummy_datamodule import DummyDataModule
import argparse


def test(model_path):
    model = DummyModel.load_from_checkpoint(model_path)

    config = model.config
    config.batch_size = 1

    model.eval()

    datamodule = DummyDataModule(config)
    datamodule.prepare_data()
    datamodule.setup("test")

    trainer = pl.Trainer()
    predictions = trainer.predict(model, datamodule.test_dataloader())

    y = [y for y, _ in predictions]
    y_hat = [y for _, y in predictions]

    y = torch.cat(y, dim=0).view(-1)
    y_hat = torch.cat(y_hat, dim=0).view(-1)

    print(y)
    print(y_hat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/nfs/ada/ferraro/users/sroydip1/dummy/checkpoints/",
        help="Checkpoint directory",
    )
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
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

    test(model_path)
