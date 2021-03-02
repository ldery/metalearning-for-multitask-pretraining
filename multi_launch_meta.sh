#!/bin/bash
#
#SBATCH --job-name=varyBatchSz
#SBATCH -e /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/meta/varyBatchSz.err
#SBATCH -o /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/meta/varyBatchSz.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=ldery@andrew.cmu.edu



# Adam. Vary Met-Head Regularization
./run4.sh softmax 1 8 8 4 1e-3 Adam metaSmallData 1
./run4.sh softmax 1 8 8 4 1e-3 Adam metaSmallData 100
./run4.sh softmax 1 8 8 4 1e-3 Adam metaSmallData 500

# # SGD
# ./run4.sh linear  0 3e-2 Adam selfsupervised/SGD.3e-2
# ./run4.sh softmax 0 3e-2 Adam selfsupervised/SGD.3e-2
