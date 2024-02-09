import argparse

def add_program_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, required=False, default=False, help='debug?')
    parser.add_argument("--seed", type=int, default=42, help="value for reproducibility") 
    parser.add_argument("--cuda", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Use CUDA?")
    parser.add_argument("--checkpoint_dir", type=str, default="/nfs/ada/ferraro/users/sroydip1/semeval24/task1/checkpoints/", help="Checkpoint directory")
    parser.add_argument("--wandb_entity", type=str, default="gcnssdvae", help="Wandb entity")
    parser.add_argument("--wandb_project", type=str, default="sem1", help="Wandb project")
    return parser

def add_data_args(parent_parser):
    parser = parent_parser.add_argument_group("Data Config")
    parser.add_argument("--data_dir", type=str, default="./data/Track A", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size?")
    return parent_parser

def add_model_args(parent_parser):
    parser = parent_parser.add_argument_group("Model Config")
    parser.add_argument("--exp_name", type=str, default="sem1", help="Experiement name?", required=True)
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-distilroberta-v1", help="Model name?")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay?")
    parser.add_argument("--enc_pooling", type=str, default="mean", help="Encoder pooling")
    parser.add_argument("--enc_dropout", type=float, default=0.1, help="Encoder dropout")
    return parent_parser

def add_trainer_args(parent_parser):
    parser = parent_parser.add_argument_group("Trainer Config")
    parser.add_argument("--max_epochs", type=int, default=-1, help="Number of maximum epochs", )
    parser.add_argument("--validate_every", type=float, default=0.125, help="Number of maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--accumulate_grad_batches", type=int, default=32, help="Number of accumulation of grad batches, -1 for no accumulation")
    parser.add_argument("--overfit", type=int, default=0, help="Overfit batches")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience? -1 if no early stopping")
    parser.add_argument("--monitoring_metric", type=str, default="valid/corr", help="Monitoring metric")
    parser.add_argument("--monitoring_mode", type=str, default="max", help="Monitoring mode")
    parser.add_argument("--ddp", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use DDP?")
    return parent_parser

def get_config():
    parser = add_program_args()
    parser = add_data_args(parser)
    parser = add_model_args(parser)
    parser = add_trainer_args(parser)
    cfg    = parser.parse_args()
    return cfg