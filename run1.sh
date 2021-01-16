#!/bin/bash

auxTasks=$1
lr=$2
optim=$3
exp_fldr=$4

# expname='default_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
# echo 'Default ' $expname ' and save file is ' $expname'.txt'
# python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy default -optimizer $optim -exp-name $exp_fldr'/default/'$expname &> run_logs/$expname'.txt'


# expname='warmUD_freq_2_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
# echo 'Warm Up And Down With Freq 2 ' $expname ' and save file is ' $expname'.txt'
# python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 2 -init-val 0.0 -end-val 1.0 -optimizer $optim -exp-name $exp_fldr'/warmup_2/'$expname &> run_logs/$expname'.txt' 


# expname='warmUD_freq_10_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
# echo 'Warm Up And Down With Freq 10 ' $expname ' and save file is ' $expname'.txt'
# python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 10 -init-val 0.0 -end-val 1.0 -optimizer $optim -exp-name $exp_fldr'/warmup_10/'$expname &> run_logs/$expname'.txt'


# expname='warmUD_freq_20_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
# echo 'Warm Up And Down With Freq 20 ' $expname ' and save file is ' $expname'.txt'
# python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 20 -init-val 0.0 -end-val 1.0 -optimizer $optim -exp-name $exp_fldr'/warmup_20/'$expname &> run_logs/$expname'.txt'


# expname='alt_2_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
# echo 'Alternating With Freq 2 ' $expname ' and save file is ' $expname'.txt'
# python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 2 -optimizer $optim -exp-name $exp_fldr'/alt_2/'$expname  &> run_logs/$expname'.txt'

# expname='alt_10_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
# echo 'Alternating With Freq 10 ' $expname ' and save file is ' $expname'.txt'
# python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 10 -optimizer $optim -exp-name $exp_fldr'/alt_10/'$expname &> run_logs/$expname'.txt'


# expname='alt_20_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
# echo 'Alternating With Freq 20 ' $expname ' and save file is ' $expname'.txt'
# python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 20 -optimizer $optim -exp-name $exp_fldr'/alt_20/'$expname &> run_logs/$expname'.txt'

# Including Phase in-and-out
primstart=0
expname='phase-in-'$primstart'_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
echo 'Phase In ' $expname ' and save file is ' $expname'.txt'
python -u main.py -prim-start $primstart -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy phase_in -optimizer $optim -exp-name $exp_fldr'/phasein_10/'$expname &> run_logs/$expname'.txt'


# primstart=10
# expname='phase-in-'$primstart'_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
# echo 'Phase In ' $expname ' and save file is ' $expname'.txt'
# python -u main.py -prim-start $primstart -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy phase_in -optimizer $optim -exp-name $exp_fldr'/phasein_10/'$expname &> run_logs/$expname'.txt'

# primstart=20
# expname='phase-in-'$primstart'_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
# echo 'Phase In ' $expname ' and save file is ' $expname'.txt'
# python -u main.py -prim-start $primstart -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy phase_in -optimizer $optim -exp-name $exp_fldr'/phasein_20/'$expname &> run_logs/$expname'.txt'

# primstart=40
# expname='phase-in-'$primstart'_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
# echo 'Phase In ' $expname ' and save file is ' $expname'.txt'
# python -u main.py -prim-start $primstart -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy phase_in -optimizer $optim -exp-name $exp_fldr'/phasein_40/'$expname &> run_logs/$expname'.txt'

# # Include regular pre-training
# expname='regular-pretrain-ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
# echo 'Regular Pretraining ' $expname ' and save file is ' $expname'.txt'
# python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain -num-runs 3 -optimizer $optim -exp-name $exp_fldr'/agnostic_pretraining/'$expname -use-last-chkpt &> run_logs/$expname'.txt'