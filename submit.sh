#!/bin/bash -x
#SBATCH --account=EUHPC_E03_068
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:05:00
#SBATCH --output=/leonardo_scratch/large/userexternal/tahmed00/dock-exp/log.txt
#SBATCH --error=/leonardo_scratch/large/userexternal/tahmed00/dock-exp/error.txt

eval "$(/leonardo/home/userexternal/tahmed00/miniforge3/bin/conda shell.bash hook)"
conda activate my_env

RESULT_DIR="/leonardo_scratch/large/userexternal/tahmed00/dock-exp"
RESULT_FILE="$RESULT_DIR/result.txt"

START_TIME=$(date +%s)

python -u /leonardo_scratch/large/userexternal/tahmed00/scripts/exp-code.py > $RESULT_FILE

END_TIME=$(date +%s)
TIME_TAKEN=$((END_TIME - START_TIME))

echo "Time taken: $TIME_TAKEN seconds" >> $RESULT_DIR/log.txt
