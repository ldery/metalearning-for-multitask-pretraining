#!/bin/bash

upAlg=$1
auxTasks=$2
split='val'
init='rand'
nruns=3

sgd_arr=(1e-3 3e-4)
weight_arr=(5e-2 1e-2)

for c in "${sgd_arr[@]}"
do
	for k in "${weight_arr[@]}"
	do
		lr=$c
		expname=$upAlg'.meta.'$split'.w_lr='$k'_sgd_lr='$c'_lr='$lr'_ntasks='$auxTasks'.'$init
		echo 'Performing on '$expname
		python -u main.py -num-aux-tasks $auxTasks -alpha-update-algo $upAlg -mode meta -num-runs $nruns -exp-name $expname -meta-lr-weights $k  -meta-lr-sgd $c -meta-split $split -lr $lr  &> run_logs/$expname'.txt'
	done
done
