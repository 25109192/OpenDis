#!/bin/bash
#SBATCH --partition=comp
#SBATCH --job-name=load
#SBATCH --nodes=1           # Request 1 node
#SBATCH --ntasks=1          # Total number of tasks
#SBATCH --cpus-per-task=48  # Number of CPU cores per task
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL

module purge
module load miniforge/25.3.1
source activate opendis_CPU
#export OMP_NUM_THREADS=48
python test_frank_read_src.py
# python bcc_Ta_5um_3e3.py 54000    # to restart from a specific checkpoint