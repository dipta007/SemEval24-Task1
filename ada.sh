#!/bin/bash

#SBATCH -D .
#SBATCH --job-name="1SWEEP"
#SBATCH --output=run/1SWEEP.log
#SBATCH --error=run/1SWEEP.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=40000
#SBATCH --time=240:00:00
#SBATCH --constraint=rtx_6000                   # NULL (12GB), rtx_6000 (24GB), rtx_8000 (48GB)

v=$(git status --porcelain | wc -l)
if [[ $v -gt 10 ]]; then
    echo "Error: uncommited changes" >&2
    exit 1
else
    echo "Success: No uncommited changes"
    $@
fi