# SemEval 2024 Task 1

## CMDS:
1. `srun --mem=40000 --time=24:00:00 --gres=gpu:1 --pty --constraint=rtx_8000 bash`


## Sweep on SLURM
1. Edit `sweep.yaml` file accordingly: [doc](https://docs.wandb.ai/guides/sweeps)
2. Run `wandb sweep sweep.yaml`
3. Get the sweep ID
4. Run `./sweep.sh $CNT $ID` where `$CNT` is the number of runs and `$ID` is the sweep ID
5. Don't change anything while sweep is running, otherwise it will be run on different code.