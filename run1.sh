#!/bin/bash

auxTasks=$1
lr=$2
optim=$3
exp_fldr=$4
ftbsz=$5
bsz=$6
dataFrac=$7
bntype=$8

# MAKE THE APPROPRIATE RUN FLDR
mkdir -p run_logs/'dataFrac-'$dataFrac

expname='dataFrac-'$dataFrac'/default_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype
echo 'Default ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy default -optimizer $optim -exp-name $exp_fldr'/default/'$expname -train-epochs 200 -patience 200  -bn-type $bntype &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


expname='dataFrac-'$dataFrac'/warmUD_freq_.05_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype
echo 'Warm Up And Down With Freq .05 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 5 -init-val 0.0 -end-val 1.0 -optimizer $optim -exp-name $exp_fldr'/warmup_.05/'$expname -train-epochs 200 -patience 200 -bn-type $bntype  &> run_logs/$expname'.txt' 
tail -n 5 run_logs/$expname'.txt'


expname='dataFrac-'$dataFrac'/warmUD_freq_.2_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype
echo 'Warm Up And Down With Freq  .2' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 40 -init-val 0.0 -end-val 1.0 -optimizer $optim -exp-name $exp_fldr'/warmup_.2/'$expname -train-epochs 200 -patience 200 -bn-type $bntype  &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


expname='dataFrac-'$dataFrac'/warmUD_freq_.5_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype
echo 'Warm Up And Down With Freq .5 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy warm_up_down -alt-freq 100 -init-val 0.0 -end-val 1.0 -optimizer $optim -exp-name $exp_fldr'/warmup_.5/'$expname -train-epochs 200 -patience 200 -bn-type $bntype  &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


expname='dataFrac-'$dataFrac'/alt_.05_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype
echo 'Alternating With Freq .05 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 5 -optimizer $optim -exp-name $exp_fldr'/alt_.05/'$expname -train-epochs 200 -patience 200 -bn-type $bntype  &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


expname='dataFrac-'$dataFrac'/alt_.2_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype
echo 'Alternating With Freq .2 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 40 -optimizer $optim -exp-name $exp_fldr'/alt_.2/'$expname -train-epochs 200 -patience 200 -bn-type $bntype  &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


expname='dataFrac-'$dataFrac'/alt_.5_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype
echo 'Alternating With Freq .5 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy alt -alt-freq 100 -optimizer $optim -exp-name $exp_fldr'/alt_.5/'$expname -train-epochs 200 -patience 200 -bn-type $bntype  &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'

# Including Phase in-and-out

primstart=5
expname='dataFrac-'$dataFrac'/phase-in-'$primstart'_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype
echo 'Phase In ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -prim-start $primstart -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy phase_in -optimizer $optim -exp-name $exp_fldr'/phasein_.05/'$expname -train-epochs 200 -patience 200 -bn-type $bntype  &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


primstart=40
expname='dataFrac-'$dataFrac'/phase-in-'$primstart'_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype
echo 'Phase In ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -prim-start $primstart -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy phase_in -optimizer $optim -exp-name $exp_fldr'/phasein_.2/'$expname -train-epochs 200 -patience 200 -bn-type $bntype  &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


primstart=100
expname='dataFrac-'$dataFrac'/phase-in-'$primstart'_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype
echo 'Phase In ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -prim-start $primstart -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs 3 -weight-strgy phase_in -optimizer $optim -exp-name $exp_fldr'/phasein_.5/'$expname -train-epochs 200 -patience 200 -bn-type $bntype  &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


# Include regular pre-training
expname='dataFrac-'$dataFrac'/regular-pretrain-ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype
echo 'Regular Pretraining ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain -num-runs 3 -optimizer $optim -exp-name $exp_fldr'/agnostic_pretraining/'$expname -use-last-chkpt -train-epochs 200 -patience 200 -bn-type $bntype &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


echo 'DONE RUNNING - CAN SHUT OFF NOW'
