#!/bin/bash
#
#SBATCH --job-name=static_baselines_smalldata
#SBATCH -e /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/smalldata.static.err
#SBATCH -o /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/smalldata.static.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=ldery@andrew.cmu.edu


# echo './run_selfsupervised.long.sh 0 2e-5 Adam selfsupervised/2e-5\n'
# ./run_selfsupervised.long.sh 0 2e-5 Adam selfsupervised/2e-5

# echo './run_selfsupervised.long.sh 0 2e-5 Adam selfsupervised/1e-4\n'
# ./run_selfsupervised.long.sh 0 1e-4 Adam selfsupervised/1e-4

# echo './run_selfsupervised.long.sh 0 5e-4 Adam selfsupervised/5e-4\n'
# ./run_selfsupervised.long.sh 0 1e-3 Adam selfsupervised/5e-4

# echo './run1.sh 2 1e-2 Adam\n'
# ./run1.sh 2 1e-1 SGD aux_2_tasks

echo './run1.sh 1 1e-4 Adam smallData_static\n'
./run1.sh 1 1e-4 Adam smallData_static

echo './run1.sh 1 1e-3 Adam smallData_static\n'
./run1.sh 1 1e-3 Adam smallData_static

echo './run1.sh 1 5e-5 Adam smallData_static\n'
./run1.sh 1 5e-5 Adam smallData_static

# echo './run1.sh 10 5e-3\n'
# ./run1.sh 10 5e-3
# echo './run1.sh 10 1e-3\n'
# ./run1.sh 10 1e-3
# echo './run1.sh 10 5e-4\n'
# ./run1.sh 10 5e-4