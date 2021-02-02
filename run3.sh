#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python -u main.py -num-aux-tasks 1 -alpha-update-algo softmax -mode meta -num-runs 1 -exp-name nolook_ablation/medium_mammals/stepped_lr/softmax.1.0 -meta-lr-weights 1e-2  -meta-lr-sgd 0 -meta-split val -lr 1e-1 -optimizer SGD -patience 200 -train-epochs 200 -main-super-class 'medium-sized_mammals' -batch-sz 128 -meta-batch-sz 32 -prim-datafrac 1.0 -use-lr-scheduler

CUDA_VISIBLE_DEVICES=2 python -u main.py -num-aux-tasks 1 -alpha-update-algo softmax -mode meta -num-runs 1 -exp-name nolook_ablation/medium_mammals/stepped_lr/softmax.0.5 -meta-lr-weights 1e-2  -meta-lr-sgd 0 -meta-split val -lr 1e-1 -optimizer SGD -patience 200 -train-epochs 200 -main-super-class 'medium-sized_mammals' -batch-sz 128 -meta-batch-sz 32 -prim-datafrac 0.5 -use-lr-scheduler

CUDA_VISIBLE_DEVICES=2 python -u main.py -num-aux-tasks 1 -alpha-update-algo softmax -mode meta -num-runs 1 -exp-name nolook_ablation/medium_mammals/stepped_lr/softmax.0.1 -meta-lr-weights 1e-2  -meta-lr-sgd 0 -meta-split val -lr 1e-1 -optimizer SGD -patience 200 -train-epochs 200 -main-super-class 'medium-sized_mammals' -batch-sz 64 -meta-batch-sz 16 -prim-datafrac 0.1 -use-lr-scheduler



# CUDA_VISIBLE_DEVICES=2 python -u main.py -num-aux-tasks 2 -alpha-update-algo softmax -mode meta -num-runs 1 -exp-name nolook_ablation/val_is_meta/small_lr.softmax -meta-lr-weights 5e-2  -meta-lr-sgd 5e-3 -meta-split val -lr 5e-3 -optimizer SGD -patience 200 -train-epochs 200

# CUDA_VISIBLE_DEVICES=2 python -u main.py -num-aux-tasks 2 -alpha-update-algo softmax -mode meta -num-runs 1 -exp-name nolook_ablation/train_is_meta/small_lr.softmax -meta-lr-weights 5e-2  -meta-lr-sgd 5e-3 -meta-split train -lr 5e-3 -optimizer SGD -patience 200 -train-epochs 200