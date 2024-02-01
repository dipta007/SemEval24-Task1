"""
This script contains an example how to extend an existent sentence embedding model to new languages.

Given a (monolingual) teacher model you would like to extend to new languages, which is specified in the teacher_model_name
variable. We train a multilingual student model to imitate the teacher model (variable student_model_name)
on multiple languages.

For training, you need parallel sentence data (machine translation training data). You need tab-seperated files (.tsv)
with the first column a sentence in a language understood by the teacher model, e.g. English,
and the further columns contain the according translations for languages you want to extend to.

This scripts downloads automatically the parallel sentences corpus. This corpus contains transcripts from
talks translated to 100+ languages. For other parallel data, see get_parallel_data_[].py scripts

Further information can be found in our paper:
Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation
https://arxiv.org/abs/2004.09813
"""

from sentence_transformers import SentenceTransformer, LoggingHandler, models, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
from datetime import datetime

import os
import logging
import sentence_transformers.util
import csv
import gzip
from tqdm.autonotebook import tqdm
import numpy as np
import zipfile
import io

teacher_model_name = "./sbert_sweep/sentence-t5-xl_wd_0.1_lrs_warmuplinear_warms_100"
student_model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"

max_seq_length = 512  # Student model max. lengths for inputs (number of word pieces)
train_batch_size = 64  # Batch size for training
inference_batch_size = 64  # Batch size at inference
max_sentences_per_language = 500000  # Maximum number of  parallel sentences for training
train_max_sentence_length = 128  # Maximum length (characters) for parallel training sentences

num_epochs = 100  # Train for x epochs
num_warmup_steps = 10000  # Warumup steps

num_evaluation_steps = 1000  # Evaluate performance after every xxxx steps
dev_sentences = 1000  # Number of parallel sentences to be used for development


# Define the language codes you would like to extend the model to
source_languages = set(["en"])  # Our teacher model accepts English (en) sentences
target_languages = set(
    # all needed: ['arq', 'amh', 'am', 'ha', 'hau', 'kin', 'rw', 'mr', 'ary', 'es', 'es_*', 'te']
    # needed: ['arq', 'am', 'ha', 'rw', 'mr', 'ary', 'es', 'te']
    # available:
    ["arq", "am", "ha", "mr", "es", "te"]
    # ["de", "es", "it", "fr", "ar", "tr"]
)  # We want to extend the model to these new languages. For language codes, see the header of the train file

# arq = qed, ted, tatoeba
# am = ted, tatoeba,
# amh = qed
# ha = ted, tatoeba,
# hau = qed
# kin = qed
# rw = tatoeba
# mr = qed, ted, tatoeba
# ary = tatoeba
# es = qed, ted, tatoeba
# te = qed, ted, tatoeba

output_path = "./sbert_sweep/student/pretrain/"


# This function downloads a corpus if it does not exist
def download_corpora(filepaths):
    if not isinstance(filepaths, list):
        filepaths = [filepaths]

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(filepath, "does not exists. Try to download from server")
            filename = os.path.basename(filepath)
            url = "https://sbert.net/datasets/" + filename
            sentence_transformers.util.http_get(url, filepath)


# Here we define train train and dev corpora
train_folder = "./data/student/combined/"
train_files = []
for source_language in source_languages:
    for target_language in target_languages:
        train_files.append(os.path.join(train_folder, "train." + source_language + "-" + target_language + ".tsv.gz"))

dev_folder = "./data/student/combined/"
dev_files = []
for source_language in source_languages:
    for target_language in target_languages:
        dev_files.append(os.path.join(dev_folder, "dev." + source_language + "-" + target_language + ".tsv.gz"))

sts_corpus = "./data/student/sts2017-extended.zip"


######## Start the extension of the teacher model to multiple languages ########
print("Load teacher model")
teacher_model = SentenceTransformer(teacher_model_name)


print("Create student model from scratch")
word_embedding_model = models.Transformer(student_model_name, max_seq_length=max_seq_length)
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


###### Read Parallel Sentences Dataset ######
train_data = ParallelSentencesDataset(
    student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=True
)
for train_file in train_files:
    train_data.load_data(
        train_file, max_sentences=max_sentences_per_language, max_sentence_length=train_max_sentence_length
    )

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=student_model)


#### Evaluate cross-lingual performance on different tasks #####
evaluators = []  # evaluators has a list of different evaluator classes we call periodically
evaluators_name = []
for dev_file in dev_files:
    print("Create evaluator for " + dev_file)
    src_sentences = []
    trg_sentences = []
    with gzip.open(dev_file, "rt", encoding="utf8") as fIn:
        for line in tqdm(fIn, desc="Sentences"):
            splits = line.strip().split("\t")
            if splits[0] != "" and splits[1] != "":
                src_sentences.append(splits[0])
                trg_sentences.append(splits[1])

    # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
    print("Create MSE Evaluator for " + dev_file)
    dev_mse = evaluation.MSEEvaluator(
        src_sentences,
        trg_sentences,
        name=os.path.basename(dev_file),
        teacher_model=teacher_model,
        batch_size=inference_batch_size,
    )
    evaluators.append(dev_mse)
    evaluators_name.append(f"MSE/{os.path.basename(dev_file)}")

    # TranslationEvaluator computes the embeddings for all parallel sentences. It then check if the embedding of source[i] is the closest to target[i] out of all available target sentences
    print("Create Translation Evaluator for " + dev_file)
    dev_trans_acc = evaluation.TranslationEvaluator(
        src_sentences, trg_sentences, name=os.path.basename(dev_file), batch_size=inference_batch_size
    )
    evaluators.append(dev_trans_acc)
    evaluators_name.append(f"TransAcc/{os.path.basename(dev_file)}")


##### Read cross-lingual Semantic Textual Similarity (STS) data ####
LANGS = ["amh", "arq", "ary", "eng", "esp", "hau", "kin", "mar", "tel"]
for lang in LANGS:
    print("Create STS evaluator for " + lang)
    sentences1 = []
    sentences2 = []
    scores = []
    df = pd.read_csv(f"./data/Track A/{lang}/val_student.tsv", sep="\t")
    pairidset = set()
    for _, row in df.iterrows():
        pairid = row["PairID"]
        if pairid not in pairidset:
            sentences1.append(row["t1_tgt"])
            sentences2.append(row["t2_tgt"])
            scores.append(row["score"])
            pairidset.add(pairid)

    # semantic similarity evaluation
    sts_evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    evaluators.append(sts_evaluator)
    evaluators_name.append(f"STS/{lang}")


def log_metrics_to_wandb(scores):
    mse = []
    trans_acc = []
    sts = []
    log_dict = {}
    for e_name, score in zip(evaluators_name, scores):
        print(f"valid/{e_name}: {score}")
        log_dict[f"valid/{e_name}"] = score
        if e_name.startswith("MSE"):
            mse.append(score)
        if e_name.startswith("TransAcc"):
            trans_acc.append(score)
        if e_name.startswith("STS"):
            sts.append(score)

    print(f"valid/mean: {np.mean(scores)}")
    print(f"valid/mse: {np.mean(mse)}")
    print(f"valid/trans: {np.mean(trans_acc)}")
    print(f"valid/corr: {np.mean(sts)}")
    log_dict[f"valid/mean"] = np.mean(scores)
    log_dict[f"valid/mse"] = np.mean(mse)
    log_dict[f"valid/trans"] = np.mean(trans_acc)
    log_dict[f"valid/corr"] = np.mean(sts)
    # wandb.log(log_dict)

    return np.mean(scores)


# Train the model
print("Start training")
student_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: log_metrics_to_wandb(scores)),
    epochs=num_epochs,
    warmup_steps=num_warmup_steps,
    evaluation_steps=num_evaluation_steps,
    output_path=output_path,
    save_best_model=True,
    optimizer_params={"lr": 2e-5, "eps": 1e-6},
)
