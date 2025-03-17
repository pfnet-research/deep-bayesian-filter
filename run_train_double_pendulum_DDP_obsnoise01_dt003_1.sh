#!/bin/bash
set -eu

#pip install kornia
#pip install filterpy
#pip install torchviz

WD=$(cd $(dirname $0); pwd)
cd $WD

cond=1
OUTDIR="${WD}/doublependulum/DDP_noise01_dt003/cond${cond}"
mkdir -p ${OUTDIR}

rm -f "${OUTDIR}/trainloss_integral.txt"
rm -f "${OUTDIR}/trainloss_KL.txt"
rm -f "${OUTDIR}/testloss_integral.txt"
rm -f "${OUTDIR}/testloss_KL.txt"
rm -f "${OUTDIR}/log_sysnoise.txt"
rm -f "${OUTDIR}/log_obsnoise.txt"
rm -f "${OUTDIR}/log_concentration_periodic.txt"

export PYTHONPATH=$PWD
python3 train_double_pendulum_DDP.py --config config/DBF/doublependulum/DDP_obsnoise01_dt003_cond${cond}.yaml
