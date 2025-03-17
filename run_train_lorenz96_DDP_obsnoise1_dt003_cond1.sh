#!/bin/bash

#pip install kornia
#pip install filterpy
#pip install torchviz

WD=$(cd $(dirname $0); pwd)
cd $WD

cond=1
OUTDIR="${WD}/Lorenz96_physspace_complete/DDP_obsnoise1_dt003/cond${cond}"
mkdir -p ${OUTDIR}

export PYTHONPATH=$PWD
python train_Lorenz96_DDP.py --config config/DBF/Lorenz96/DDP_obsnoise1_dt003_cond${cond}.yaml
