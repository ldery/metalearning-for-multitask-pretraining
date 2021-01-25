#!/bin/bash

upAlg=$1
auxTasks=$2
blr=$3
lr=$4
slr=$5
optim=$6
exp_fldr=$7
split='val'
init='rand'
nruns=3


sub='selfsup'

sgd_arr=($blr $lr $slr)
weight_arr=(5e-2)

for c in "${sgd_arr[@]}"
do
	for k in "${weight_arr[@]}"
	do
		lr=$c
		expname=$upAlg'.meta.'$split'.w_lr='$k'_sgd_lr='$c'_lr='$lr'_ntasks='$auxTasks'.'$init'.'$optim
		echo 'Performing on '$expname
		if [[ "$exp_fldr" == *"$sub"* ]]; then
			python -u main.py -num-aux-tasks $auxTasks -alpha-update-algo $upAlg -mode meta -num-runs $nruns -exp-name $exp_fldr'/meta/'$expname -meta-lr-weights $k  -meta-lr-sgd $c -meta-split $split -lr $lr -optimizer $optim -use-crop -use-rotation &> run_logs/$expname'.txt'
			tail -n 5 run_logs/$expname'.txt'
		else
			python -u main.py -num-aux-tasks $auxTasks -alpha-update-algo $upAlg -mode meta -num-runs $nruns -exp-name $exp_fldr'/meta/'$expname -meta-lr-weights $k  -meta-lr-sgd $c -meta-split $split -lr $lr -optimizer $optim &> run_logs/$expname'.txt'
			tail -n 5 run_logs/$expname'.txt'
		fi
		
	done
done
