#!/bin/bash

echo 'Alternating With Freq 5'
python -u main.py -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 5 -exp-name alt_5_norm &> run_logs/alt_5_norm.txt

echo 'Alternating With Freq 2'
python -u main.py -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 2 -exp-name alt_2_norm &> run_logs/alt_2_norm.txt
