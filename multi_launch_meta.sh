#!/bin/bash
#
#SBATCH --job-name=meta_runs_selfsupervised
#SBATCH -e /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/selfsupervised.meta.err
#SBATCH -o /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/selfsupervised.meta.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=ldery@andrew.cmu.edu

# Adam
./run4.sh linear  0 4e-5 2e-5 1e-5 Adam selfsupervised/2e-5
./run4.sh softmax 0 4e-5 2e-5 1e-5 Adam selfsupervised/2e-5

# # SGD
# ./run4.sh linear  0 1e-1 SGD selfsupervised/SGD.1e-1
# ./run4.sh softmax 0 1e-1 SGD selfsupervised/SGD.1e-1

# Adam
./run4.sh linear  0 2e-4 1e-4 5e-5 Adam selfsupervised/1e-4
./run4.sh softmax 0 2e-4 1e-4 5e-5 Adam selfsupervised/1e-4

# # SGD
# ./run4.sh linear  0 3e-2 Adam selfsupervised/SGD.3e-2
# ./run4.sh softmax 0 3e-2 Adam selfsupervised/SGD.3e-2
