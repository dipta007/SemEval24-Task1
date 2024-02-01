from sentence_transformers import SentenceTransformer, models, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
import wandb


LANGS = [
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

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--teacher_model",
    type=str,
    # default="./sbert_sweep/all-roberta-large-v1_wd_0.001_lrs_warmupcosinewithhardrestarts_warms_1000",
    default="./sbert_sweep/sentence-t5-xl_wd_0.1_lrs_warmuplinear_warms_100",
)
argparser.add_argument(
    "--student_model",
    type=str,
    default="xlm-roberta-large"
    # default="Soonhwan-Kwon/xlm-roberta-xlarge"
    # default="facebook/xlm-roberta-xl",
)
argparser.add_argument(
    "--translation_model",
    type=str,
    default="all"
    # default="Soonhwan-Kwon/xlm-roberta-xlarge"
    # default="facebook/xlm-roberta-xl",
)
argparser.add_argument("--lr", type=float, default=2e-5)
argparser.add_argument("--batch_size", type=int, default=256)
argparser.add_argument("--epochs", type=int, default=400)
argparser.add_argument("--validate_every", type=int, default=130)
argparser.add_argument("--lr_scheduler", type=str, default="constantlr")
argparser.add_argument("--warmup_steps", type=int, default=100)
argparser.add_argument("--weight_decay", type=float, default=0.00)
argparser.add_argument("--seed", type=int, default=42)


args = argparser.parse_args()

output_path = f"./sbert_sweep/student/{args.student_model}"


######## Start the extension of the teacher model to multiple languages ########
print("Load teacher model")
teacher_model = SentenceTransformer(args.teacher_model)


print("Create student model from scratch")
word_embedding_model = models.Transformer(args.student_model)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


###### Read Parallel Sentences Dataset ######
train_data = ParallelSentencesDataset(
    student_model=student_model, teacher_model=teacher_model, batch_size=args.batch_size, use_embedding_cache=True
)
for lang in LANGS:
    print("Load training data for", lang)
    train_data.load_data(f"./data/Track A/{lang}/train_student.tsv")


train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
train_loss = losses.MSELoss(model=student_model)


#### Evaluate cross-lingual performance on different tasks #####
evaluators = []
evaluators_name = []
for lang in LANGS:
    print("Create evaluator for " + lang)
    src_sentences = []
    tgt_sentences = []
    sentences1 = []
    sentences2 = []
    scores = []
    df = pd.read_csv(f"./data/Track A/{lang}/val_student.tsv", sep="\t")
    pairidset = set()
    for _, row in df.iterrows():
        src_sentences.append(row["t1_src"])
        tgt_sentences.append(row["t1_tgt"])

        src_sentences.append(row["t2_src"])
        tgt_sentences.append(row["t2_tgt"])

        pairid = row["PairID"]
        if pairid not in pairidset:
            sentences1.append(row["t1_tgt"])
            sentences2.append(row["t2_tgt"])
            scores.append(row["score"])
            pairidset.add(pairid)

    # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
    dev_mse = evaluation.MSEEvaluator(
        src_sentences,
        tgt_sentences,
        teacher_model=teacher_model,
        batch_size=args.batch_size,
    )
    evaluators.append(dev_mse)
    evaluators_name.append(f"MSE/{lang}")

    # TranslationEvaluator computes the embeddings for all parallel sentences. It then check if the embedding of source[i] is the closest to target[i] out of all available target sentences
    dev_trans_acc = evaluation.TranslationEvaluator(src_sentences, tgt_sentences, name=lang, batch_size=args.batch_size)
    evaluators.append(dev_trans_acc)
    evaluators_name.append(f"TransAcc/{lang}")

    # semantic similarity evaluation
    sts_evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    evaluators.append(sts_evaluator)
    evaluators_name.append(f"STS/{lang}")


wandb.login()
wandb.init(
    entity="gcnssdvae",
    project="sem1",
    tags=["sbert", "teacher_student"],
    name=args.student_model,
    config={
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "translation_model": args.translation_model,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "validate_every": args.validate_every,
        "lr_scheduler": args.lr_scheduler,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    },
)
wandb.define_metric("valid/corr", summary="max")


def log_metrics_to_wandb(scores, step):
    mse = []
    trans_acc = []
    sts = []
    log_dict = {}
    for e_name, score in zip(evaluators_name, scores):
        # print(f"valid/{e_name}: {score}")
        log_dict[f"valid/{e_name}"] = score
        if e_name.startswith("MSE"):
            mse.append(score)
        if e_name.startswith("TransAcc"):
            trans_acc.append(score)
        if e_name.startswith("STS"):
            sts.append(score)

    # print(f"valid/step: {step}")
    # print(f"valid/mean: {np.mean(scores)}")
    # print(f"valid/mse: {np.mean(mse)}")
    # print(f"valid/trans: {np.mean(trans_acc)}")
    # print(f"valid/corr: {np.mean(sts)}")
    log_dict[f"valid/step"] = step
    log_dict[f"valid/mean"] = np.mean(scores)
    log_dict[f"valid/mse"] = np.mean(mse)
    log_dict[f"valid/trans"] = np.mean(trans_acc)
    log_dict[f"valid/corr"] = np.mean(sts)
    wandb.log(log_dict)
    
    step += 1

    return np.mean(scores)


step = 1
# Train the model
student_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluation.SequentialEvaluator(
        evaluators, main_score_function=lambda scores: log_metrics_to_wandb(scores, step)
    ),
    optimizer_params={"lr": args.lr},
    scheduler=args.lr_scheduler,
    warmup_steps=args.warmup_steps,
    weight_decay=args.weight_decay,
    evaluation_steps=args.validate_every,
    output_path=output_path,
    save_best_model=True,
    epochs=args.epochs,
)
