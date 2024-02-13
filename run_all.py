import os


models = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-roberta-large-v1",
]
enc_pooling = ["mean", "max", "cls"]
batch_size = [2, 4, 8]

def call(cmd):
    print(cmd)
    os.system(cmd)

for model in models:
    exp_name = f"final_model-{model.split('/')[-1]}"
    cmd = f"sbatch ada.sh python src/train.py --model_name {model} --exp_name {exp_name}"
    call(cmd)

for pooling in enc_pooling:
    exp_name = f"final_pooling-{pooling}"
    cmd = f"sbatch ada.sh python src/train.py --enc_pooling {pooling} --exp_name {exp_name}"
    call(cmd)

for bs in batch_size:
    exp_name = f"final_bs-{bs}"
    cmd = f"sbatch ada.sh python src/train.py --batch_size {bs} --exp_name {exp_name}"
    call(cmd)

cmd = f"sbatch ada.sh python src/train.py --exp_name=final"
call(cmd)