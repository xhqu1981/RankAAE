#!/bin/bash
#SBATCH -p long
#SBATCH -J opt2_m12
#SBATCH --time=1-00:00:00
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH -c 36
####SBATCH --dependency=afterany:690055

cp ../rand_2/opt_daae.rdb .

export CONDA_SHLVL=1
export CONDA_PROMPT_MODIFIER=(/hpcgpfs01/software/cfn-jupyter/software/xas_ml)
export CONDA_EXE=/sdcc/u/xiaqu/program/miniconda3/bin/conda
export PATH=/hpcgpfs01/software/cfn-jupyter/software/xas_ml/bin:$PATH
export CONDA_PREFIX=/hpcgpfs01/software/cfn-jupyter/software/xas_ml
export CONDA_PYTHON_EXE=/sdcc/u/xiaqu/program/miniconda3/bin/python
export CONDA_DEFAULT_ENV=/hpcgpfs01/software/cfn-jupyter/software/xas_ml

hn=$(hostname -s)
sed -i "s/^bind.*/bind ${hn}/" redis.conf

if [[ ! -f "opt_daae.rdb" ]]
then
    echo No existing DB file, create new DB
    redis-server redis.conf &
    optuna create-study --study-name "opt_daae" --storage "redis://${hn}:6379" --direction maximize
    redis-cli -h ${hn} shutdown
    echo Done with DB creation
    echo
fi

redis-server redis.conf &
sleep 10

srun -n ${SLURM_JOB_NUM_NODES} --ntasks-per-node 1 -c 36 opt_hyper.sh 8 ${SLURMD_NODENAME} -s -m -d ti_feff_cn_wei_spec.csv -c opt_config.yaml --fixed_params fix_config.yaml


