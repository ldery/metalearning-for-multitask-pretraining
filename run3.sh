#!/bin/bash

auxTasks=$1
lr=$2
optim=$3
exp_fldr=$4
ftbsz=$5
bsz=$6
dataFrac=$7
bntype=$8
ftlr=$9

# MAKE THE APPROPRIATE RUN FLDR
mkdir -p run_logs/'dataFrac-'$dataFrac

# Include regular pre-training
expname='dataFrac-'$dataFrac'/regular-pretrain-ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype'-uselast.False-ftlr.'$ftlr
echo 'Regular Pretraining ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-lr $ftlr -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain -num-runs 3 -optimizer $optim -exp-name $exp_fldr'/agnostic_pretraining/'$expname -train-epochs 200 -patience 200 -bn-type $bntype &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


echo 'DONE RUNNING - CAN SHUT OFF NOW'
