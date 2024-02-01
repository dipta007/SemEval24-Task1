#!/bin/bash

#SBATCH -D .
#SBATCH --job-name="steacher1"
#SBATCH --output=steacher1.log
#SBATCH --error=steacher1.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100000
#SBATCH --time=240:00:00

$@
