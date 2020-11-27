#!/bin/bash

auxTasks=$1
lr=$2


expname='default_ntasks='$auxTasks'-lr.'$lr
echo 'Default ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy default -exp-name $expname &> run_logs/$expname'.txt'


expname='warmUD_freq_2_ntasks='$auxTasks'-lr.'$lr
echo 'Warm Up And Down With Freq 2 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 2 -init-val 0.0 -end-val 1.0 -exp-name $expname &> run_logs/$expname'.txt' 


expname='warmUD_freq_5_ntasks='$auxTasks'-lr.'$lr
echo 'Warm Up And Down With Freq 5 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 5 -init-val 0.0 -end-val 1.0 -exp-name $expname &> run_logs/$expname'.txt'


expname='warmUD_freq_10_ntasks='$auxTasks'-lr.'$lr
echo 'Warm Up And Down With Freq 10 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 10 -init-val 0.0 -end-val 1.0 -exp-name $expname &> run_logs/$expname'.txt'


expname='alt_2_ntasks='$auxTasks'-lr.'$lr
echo 'Alternating With Freq 2 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 2 -exp-name $expname  &> run_logs/$expname'.txt'

expname='alt_5_ntasks='$auxTasks'-lr.'$lr
echo 'Alternating With Freq 5 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 5 -exp-name $expname &> run_logs/$expname'.txt'


expname='alt_10_ntasks='$auxTasks'-lr.'$lr
echo 'Alternating With Freq 10 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 10 -exp-name $expname &> run_logs/$expname'.txt'

# Including Phase in-and-out
