#!/bin/bash
#
#SBATCH --job-name=meta_runs
#SBATCH -e /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/meta.err
#SBATCH -o /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/meta.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=ldery@andrew.cmu.edu

# Adam Shorter
./run4.sh linear  2 3e-4 Adam aux_2_tasks/meta.shorter
./run4.sh softmax 2 3e-4 Adam aux_2_tasks/meta.shorter

# Adam Longer
./run4.sh linear  2 2e-5 Adam aux_2_tasks/meta.longer
./run4.sh softmax 2 2e-5 Adam aux_2_tasks/meta.longer