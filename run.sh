# sbatch ada.sh python src/train.py --batch_size=16 --accumulate_grad_batches=64 --enc_pooling="cls" --lr=0.00009272506169448252 --model_name="sentence-transformers/all-mpnet-base-v1" --weight_decay=0.0 --exp_name=best_sweep_1 --ddp

# sbatch ada.sh python src/train.py --batch_size=16 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003299320272943358 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --exp_name=best_sweep_2 --ddp

sbatch ada.sh python src/train.py --batch_size=16 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --validate_every=0.125 --exp_name=sweep2_all --monitoring_metric=all/valid/corr --ddp

sbatch ada.sh python src/train.py --batch_size=16 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --validate_every=0.125 --exp_name=sweep2_ary --monitoring_metric=ary/valid/corr --ddp

sbatch ada.sh python src/train.py --batch_size=16 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --validate_every=0.125 --exp_name=sweep2_eng --monitoring_metric=eng/valid/corr --ddp

sbatch ada.sh python src/train.py --batch_size=16 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --validate_every=0.125 --exp_name=sweep2_esp --monitoring_metric=esp/valid/corr --ddp

sbatch ada.sh python src/train.py --batch_size=16 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --validate_every=0.125 --exp_name=sweep2_hau --monitoring_metric=hau/valid/corr --ddp

sbatch ada.sh python src/train.py --batch_size=16 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --validate_every=0.125 --exp_name=sweep2_kin --monitoring_metric=kin/valid/corr --ddp

sbatch ada.sh python src/train.py --batch_size=16 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --validate_every=0.125 --exp_name=sweep2_mar --monitoring_metric=mar/valid/corr --ddp

sbatch ada.sh python src/train.py --batch_size=16 --accumulate_grad_batches=128 --enc_pooling="mean" --lr=0.00003 --model_name="sentence-transformers/all-mpnet-base-v2" --weight_decay=0.01 --validate_every=0.125 --exp_name=sweep2_tel --monitoring_metric=tel/valid/corr --ddp

