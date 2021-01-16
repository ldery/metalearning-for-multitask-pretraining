#!/bin/bash

upAlg=$1
auxTasks=$2
lr=$3
optim=$4
exp_fldr=$5
split='val'
init='rand'
nruns=3

sgd_arr=($lr)
weight_arr=(5e-2 1e-1)

for c in "${sgd_arr[@]}"
do
	for k in "${weight_arr[@]}"
	do
		lr=$c
		expname=$upAlg'.meta.'$split'.w_lr='$k'_sgd_lr='$c'_lr='$lr'_ntasks='$auxTasks'.'$init'.'$optim
		echo 'Performing on '$expname
		python -u main.py -num-aux-tasks $auxTasks -alpha-update-algo $upAlg -mode meta -num-runs $nruns -exp-name $exp_fldr'/meta/'$expname -meta-lr-weights $k  -meta-lr-sgd $c -meta-split $split -lr $lr -optimizer $optim &> run_logs/$expname'.txt'
	done
done
