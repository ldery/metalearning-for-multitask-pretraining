#!/bin/bash
#
#SBATCH --job-name=varyMetaLR.1
#SBATCH -e /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/meta/varyMetaLR.1.err
#SBATCH -o /home/ldery/meta4multitask_all/meta4multitask/m4m_cache/slurm_logs/meta/varyMetaLR.1.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=ldery@andrew.cmu.edu


# Adam. Vary Met-Head Regularization
# ./run4.sh softmax 1 8 8 4 1e-4 Adam metaSmallData 1
./run4.sh softmax 1 8 8 4 1e-4 Adam metaSmallData 1




# # Adam. Vary Met-Head Regularization
# ./run4.sh softmax 1 16 16 8 1e-3 Adam metaSmallData 1
# ./run4.sh softmax 1 16 16 8 1e-3 Adam metaSmallData 100
# ./run4.sh softmax 1 16 16 8 1e-3 Adam metaSmallData 500

# # SGD
# ./run4.sh linear  0 3e-2 Adam selfsupervised/SGD.3e-2
# ./run4.sh softmax 0 3e-2 Adam selfsupervised/SGD.3e-2
