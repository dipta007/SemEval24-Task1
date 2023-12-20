#!/bin/bash
#SBATCH --mail-type=ALL                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sroydip1+ada@umbc.edu       # Where to send mail
#SBATCH -D .
#SBATCH --job-name="jupyter"
#SBATCH --output=log/output/jupyter.log
#SBATCH --error=log/error/jupyter.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100000
#SBATCH --time=240:00:00
#SBATCH --constraint=rtx_8000                   # NULL (12GB), rtx_6000 (24GB), rtx_8000 (48GB)

port=8888
node=$(hostname -s)
user=$(whoami)


jupyter-notebook --no-browser --port=${port} --ip=${node}

# ssh -N -L <local_port>:<node_nodelist(g12)>:<port> <user>@<server>