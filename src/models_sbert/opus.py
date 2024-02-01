import pandas as pd
from sklearn.model_selection import train_test_split
import sentence_transformers.util
from opustools import OpusRead
import os
import tarfile
import gzip
import csv
from tqdm.autonotebook import tqdm
import os
import shutil


RANDOM_SEED = 42

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

source_languages = set(["en"])  # Our teacher model accepts English (en) sentences
target_languages = set(
    # all needed: 
    ['arq', 'amh', 'am', 'ha', 'hau', 'kin', 'rw', 'mr', 'ary', 'es', 'te']
    # needed: ['arq', 'am', 'ha', 'rw', 'mr', 'ary', 'es', 'te']
    # available:
    # ['arq', 'am', 'ha', 'mr', 'es', 'te']
    # ["de", "es", "it", "fr", "ar", "tr"]
)  # We want to extend the model to these new languages. For language codes, see the header of the train file

corpora = [
    "TED2020 v1"
]  # Corpora you want to use

output_folder = "../../data/student/"
opus_download_folder = "./opus"

# Iterator over all corpora / source languages / target languages combinations and download files
os.makedirs(output_folder, exist_ok=True)

for corpus in corpora:
    for src_lang in source_languages:
        for trg_lang in target_languages:
            corpus_folder = os.path.join(output_folder, corpus)
            os.makedirs(corpus_folder, exist_ok=True)
            output_filename = os.path.join(output_folder, corpus, "OPUS-{}-{}-train.tsv.gz".format(src_lang, trg_lang))
            if not os.path.exists(output_filename):
                print("Create:", output_filename)
                try:
                    read = OpusRead(
                        directory=corpus,
                        source=src_lang,
                        target=trg_lang,
                        write=[output_filename],
                        download_dir=opus_download_folder,
                        preprocess="raw",
                        write_mode="moses",
                        suppress_prompts=True,
                        verbose=True,
                    )
                    read.printPairs()
                except Exception as e:
                    print("error:", output_filename, e)
                    os.remove(output_filename)

shutil.rmtree("./opus/")
print("---DONE---")
