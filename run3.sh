#!/bin/bash
echo 'Meta With meta-lr-sgd 2.0'
python -u main.py -mode meta -num-runs 3 -exp-name meta_m_wgts_lr=2 -meta-lr-weights 2.0 &> run_logs/new_meta_m_wgts_lr=2.txt

echo 'Meta With Defaults'
python -u main.py -mode meta -num-runs 3 -exp-name meta &> run_logs/new_meta.txt

echo 'Meta With meta-lr-sgd 5.0'
python -u main.py -mode meta -num-runs 3 -exp-name meta_m_wgts_lr=5 -meta-lr-weights 5 &> run_logs/new_meta_m_wgts_lr=5.txt
