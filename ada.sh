#!/bin/bash

#SBATCH -D .
#SBATCH --job-name="steacher"
#SBATCH --output=run/steacher.log
#SBATCH --error=run/steacher.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=70000
#SBATCH --time=240:00:00
#SBATCH --nodelist=g12

v=$(git status --porcelain | wc -l)
if [[ $v -gt 10 ]]; then
    echo "Error: uncommited changes" >&2
    exit 1
else
    echo "Success: No uncommited changes"
    $@
fi