# sbatch ada.sh python src/train.py --batch_size=32 --accumulate_grad_batches=64 --enc_pooling="cls" --lr=0.00009272506169448252 --model_name="sentence-transformers/all-mpnet-base-v1" --weight_decay=0.0 --exp_name=best_sweep_1

# sbatch ada.sh python src/train.py --batch_size=32 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003299320272943358 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --exp_name=best_sweep_2

sbatch ada.sh python src/train.py --batch_size=32 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --exp_name=sweep2_ary --monitoring_metric=valid/ary/corr --validate_every=1.0

sbatch ada.sh python src/train.py --batch_size=32 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --exp_name=sweep2_eng --monitoring_metric=valid/eng/corr --validate_every=1.0

sbatch ada.sh python src/train.py --batch_size=32 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --exp_name=sweep2_esp --monitoring_metric=valid/esp/corr --validate_every=1.0

sbatch ada.sh python src/train.py --batch_size=32 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --exp_name=sweep2_hau --monitoring_metric=valid/hau/corr --validate_every=1.0

sbatch ada.sh python src/train.py --batch_size=32 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --exp_name=sweep2_kin --monitoring_metric=valid/kin/corr --validate_every=1.0

sbatch ada.sh python src/train.py --batch_size=32 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --exp_name=sweep2_mar --monitoring_metric=valid/mar/corr --validate_every=1.0

sbatch ada.sh python src/train.py --batch_size=32 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --exp_name=sweep2_tel --monitoring_metric=valid/tel/corr --validate_every=1.0