#!/bin/bash

upAlg=$1
auxTasks=$2
ftbsz=$3
bsz=$4
mbsz=$5
lr=$6
optim=$7
exp_fldr=$8
aReg=$9
split='val'
nruns=3
primfrac=0.05
sgdlr=0

weight_arr=(1e-1 5e-2)

for k in "${weight_arr[@]}"
do
	lr=$lr
	expname=$upAlg'.meta.'$split'-wlr.'$k'-sgdlr.'$sgdlr'-lr.'$lr'-ntasks.'$auxTasks'-optim.'$optim'-AlphaReg.'$aReg'-ftbsz.'$ftbsz'-bsz.'$bsz'-mbsz.'$mbsz
	echo 'Performing on '$expname

	python -u main.py -num-aux-tasks $auxTasks -alpha-update-algo $upAlg -mode meta -num-runs $nruns -exp-name $exp_fldr'/meta/'$expname -meta-lr-weights $k  -meta-lr-sgd $sgdlr -meta-split $split -lr $lr -optimizer $optim -patience 200 -train-epochs 200 -main-super-class 'medium-sized_mammals' -meta-reg-alpha $aReg -prim-datafrac $primfrac -ft-batch-sz $ftbsz  -batch-sz $bsz -meta-batch-sz $mbsz &> run_logs/smallDAtaMeta/$expname'.txt'
	tail -n 5 run_logs/smallDAtaMeta/$expname'.txt'
done

