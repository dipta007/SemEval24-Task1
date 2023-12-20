#!/bin/bash

#SBATCH -D .
#SBATCH --job-name="S1"
#SBATCH --output=run/S1.log
#SBATCH --error=run/S1.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH --time=240:00:00
#SBATCH --constraint=rtx_6000                   # NULL (12GB), rtx_6000 (24GB), rtx_8000 (48GB)

v=$(git status --porcelain | wc -l)
if [[ $v -gt 1 ]]; then
    echo "Error: uncommited changes" >&2
    exit 1
else
    echo "Success: No uncommited changes"
    $@
fi