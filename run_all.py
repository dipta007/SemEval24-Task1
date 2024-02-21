import os


models = [
    # "sentence-transformers/all-mpnet-base-v2",
    # "sentence-transformers/all-roberta-large-v1",
    # "sentence-transformers/all-MiniLM-L12-v2",
    # "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    # "sentence-transformers/sentence-t5-xl"
]
# enc_pooling = ["max", "cls"]
# batch_size = [4, 8]
# batch_size = [4]
# accu_grads = [16, 64, 128, 256]

def call(cmd):
    print(cmd)
    os.system(cmd)

for model in models:
    exp_name = f"final_model-{model.split('/')[-1]}"
    cmd = f"sbatch ada.sh python src/train.py --model_name {model} --exp_name {exp_name}"
    call(cmd)

# for pooling in enc_pooling:
#     exp_name = f"final_pooling-{pooling}"
#     cmd = f"sbatch ada.sh python src/train.py --enc_pooling {pooling} --exp_name {exp_name}"
#     call(cmd)

# for bs in batch_size:
#     exp_name = f"final_bs-{bs}"
#     cmd = f"sbatch ada.sh python src/train.py --batch_size {bs} --exp_name {exp_name} --accumulate_grad_batches -1"
#     call(cmd)

# for accu_grad in accu_grads:
#     exp_name = f"final_accu_grad-{accu_grad}"
#     cmd = f"sbatch ada.sh python src/train.py --accumulate_grad_batches {accu_grad} --exp_name {exp_name}"
#     call(cmd)


# cmd = f"sbatch ada.sh python src/train.py --exp_name=final"
# call(cmd)