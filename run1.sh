#!/bin/bash

auxTasks=$1
lr=$2
optim=$3
exp_fldr=$4

expname='default_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
echo 'Default ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy default -optimizer $optim -exp-name $exp_fldr'/default/'$expname -train-epochs 400 -patience 400 &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


expname='warmUD_freq_.05_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
echo 'Warm Up And Down With Freq .05 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 20 -init-val 0.0 -end-val 1.0 -optimizer $optim -exp-name $exp_fldr'/warmup_.05/'$expname -train-epochs 400 -patience 400 &> run_logs/$expname'.txt' 
tail -n 5 run_logs/$expname'.txt'


expname='warmUD_freq_.2_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
echo 'Warm Up And Down With Freq  .2' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 80 -init-val 0.0 -end-val 1.0 -optimizer $optim -exp-name $exp_fldr'/warmup_.2/'$expname -train-epochs 400 -patience 400 &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


expname='warmUD_freq_.5_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
echo 'Warm Up And Down With Freq .5 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 200 -init-val 0.0 -end-val 1.0 -optimizer $optim -exp-name $exp_fldr'/warmup_.5/'$expname -train-epochs 400 -patience 400 &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


expname='alt_.05_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
echo 'Alternating With Freq .05 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 20 -optimizer $optim -exp-name $exp_fldr'/alt_.05/'$expname -train-epochs 400 -patience 400 &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


expname='alt_.2_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
echo 'Alternating With Freq .2 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 80 -optimizer $optim -exp-name $exp_fldr'/alt_.2/'$expname -train-epochs 400 -patience 400 &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


expname='alt_.5_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
echo 'Alternating With Freq .5 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 200 -optimizer $optim -exp-name $exp_fldr'/alt_.5/'$expname -train-epochs 400 -patience 400 &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'

# Including Phase in-and-out

primstart=20
expname='phase-in-'$primstart'_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
echo 'Phase In ' $expname ' and save file is ' $expname'.txt'
python -u main.py -prim-start $primstart -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy phase_in -optimizer $optim -exp-name $exp_fldr'/phasein_.05/'$expname -train-epochs 400 -patience 400 &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


primstart=80
expname='phase-in-'$primstart'_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
echo 'Phase In ' $expname ' and save file is ' $expname'.txt'
python -u main.py -prim-start $primstart -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy phase_in -optimizer $optim -exp-name $exp_fldr'/phasein_.2/'$expname -train-epochs 400 -patience 400 &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


primstart=200
expname='phase-in-'$primstart'_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
echo 'Phase In ' $expname ' and save file is ' $expname'.txt'
python -u main.py -prim-start $primstart -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy phase_in -optimizer $optim -exp-name $exp_fldr'/phasein_.5/'$expname -train-epochs 400 -patience 400 &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


# Include regular pre-training
expname='regular-pretrain-ntasks='$auxTasks'-lr.'$lr'-optim.'$optim
echo 'Regular Pretraining ' $expname ' and save file is ' $expname'.txt'
python -u main.py -lr $lr -num-aux-tasks $auxTasks -mode pretrain -num-runs 3 -optimizer $optim -exp-name $exp_fldr'/agnostic_pretraining/'$expname -use-last-chkpt -train-epochs 400 -patience 400 &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


echo 'DONE RUNNING - CAN SHUT OFF NOW'