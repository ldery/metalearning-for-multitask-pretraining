#!/bin/bash

echo 'Warm Up And Down With Freq 5'
python -u main.py -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 5 -init-val 0.0 -end-val 1.0 -exp-name warmUD_freq_5_norm &> run_logs/warmUD_freq_5_norm.txt

echo 'Warm Up And Down With Freq 2'
python -u main.py -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 2 -init-val 0.0 -end-val 1.0 -exp-name warmUD_freq_2_norm &> run_logs/warmUD_freq_2_norm.txt

echo 'Default'
python -u main.py -mode pretrain_w_all -num-runs 3 -weight-strgy default -exp-name default &> run_logs/default_norm.txt