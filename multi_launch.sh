#!/bin/bash
#
#SBATCH --job-name=static_baselines_longer
#SBATCH -e /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/static.longer.err
#SBATCH -o /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/static.longer.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=ldery@andrew.cmu.edu


echo './run1.sh 2 2e-5 Adam aux_2_tasks/longer_runs\n'
./run1.sh 2 2e-5 Adam aux_2_tasks/longer_runs

# echo './run1.sh 2 1e-2 Adam\n'
# ./run1.sh 2 1e-1 SGD aux_2_tasks

# echo './run1.sh 4 5e-3\n'
# ./run1.sh 4 5e-3

# echo './run1.sh 4 1e-3\n'
# ./run1.sh 4 1e-3

# echo './run1.sh 10 5e-3\n'
# ./run1.sh 10 5e-3
# echo './run1.sh 10 1e-3\n'
# ./run1.sh 10 1e-3
# echo './run1.sh 10 5e-4\n'
# ./run1.sh 10 5e-4