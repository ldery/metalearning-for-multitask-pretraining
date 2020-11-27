#!/bin/bash


sgd_arr=(1e-1 1e-2 1e-3)
weight_arr=(1e-2 1e-3 1e-4)

for c in "${sgd_arr[@]}"
do
	for k in "${weight_arr[@]}"
	do
		expname='meta.w_lr='$k'_sgd_lr='$c
		python -u main.py -mode meta -num-runs 3 -exp-name $expname -meta-lr-weights $k  -meta-lr-sgd $c &> run_logs/$expname'.txt'
	done
done

# python -u main.py -mode meta -num-runs 3 -exp-name meta_m_wgts_lr=2 -meta-lr-weights 2.0 &> run_logs/new_meta_m_wgts_lr=2.txt

# echo 'Meta With Defaults'
# python -u main.py -mode meta -num-runs 3 -exp-name meta &> run_logs/new_meta.txt

# echo 'Meta With meta-lr-sgd 5.0'
# python -u main.py -mode meta -num-runs 3 -exp-name meta_m_wgts_lr=5 -meta-lr-weights 5 &> run_logs/new_meta_m_wgts_lr=5.txt
