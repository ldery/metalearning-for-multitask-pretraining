#!/bin/bash


#SBATCH --job-name=static_baselines_smalldata.3
#SBATCH -e /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/smalldata.static.3.err
#SBATCH -o /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/smalldata.static.3.out
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

# echo './run1.sh 1 1e-4 Adam smallData_static\n'
# ./run1.sh 1 1e-4 Adam smallData_static

# # VARY BATCH-NORM TYPE
# echo './run1.sh 1 1e-3 Adam smallData_static 16 16 0.05 separate'
# ./run1.sh 1 1e-3 Adam smallData_static 16 16 0.05 separate

# echo './run1.sh 1 1e-3 Adam smallData_static 16 16 0.05 grouped'
# ./run1.sh 1 1e-3 Adam smallData_static 16 16 0.05 grouped


# # Vary Learning Rate
# echo './run1.sh 1 1e-4 Adam smallData_static 16 16 0.05 separate'
# ./run1.sh 1 1e-4 Adam smallData_static 16 16 0.05 separate

# echo './run1.sh 1 1e-4 Adam smallData_static 16 16 0.05 grouped'
# ./run1.sh 1 1e-4 Adam smallData_static 16 16 0.05 grouped


# # VARY BATCH-SIZE TYPE
echo './run1.sh 1 3e-3 Adam smallData_static 8 8 0.05 separate'
./run1.sh 1 3e-3 Adam smallData_static 8 8 0.05 separate

echo './run1.sh 1 3e-3 Adam smallData_static 8 8 0.05 grouped'
./run1.sh 1 3e-3 Adam smallData_static 8 8 0.05 grouped