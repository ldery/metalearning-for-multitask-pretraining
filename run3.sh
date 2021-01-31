#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python -u main.py -num-aux-tasks 2 -alpha-update-algo softmax -mode meta -num-runs 1 -exp-name nolook_ablation/val_is_meta/small_lr.softmax -meta-lr-weights 5e-2  -meta-lr-sgd 5e-3 -meta-split val -lr 5e-3 -optimizer SGD -patience 200 -train-epochs 200

CUDA_VISIBLE_DEVICES=2 python -u main.py -num-aux-tasks 2 -alpha-update-algo softmax -mode meta -num-runs 1 -exp-name nolook_ablation/train_is_meta/small_lr.softmax -meta-lr-weights 5e-2  -meta-lr-sgd 5e-3 -meta-split train -lr 5e-3 -optimizer SGD -patience 200 -train-epochs 200