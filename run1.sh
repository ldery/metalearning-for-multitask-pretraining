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
dropRate=${10}
nruns=5

# MAKE THE APPROPRIATE RUN FLDR
mkdir -p run_logs/'dataFrac-'$dataFrac

expname='dataFrac-'$dataFrac'/default_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype'-dp.'$dropRate
echo 'Default ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs $nruns -weight-strgy default -optimizer $optim -exp-name $exp_fldr'/default/'$expname -train-epochs 100 -patience 30  -bn-type $bntype -dropRate $dropRate -ft-dropRate $dropRate &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


expname='dataFrac-'$dataFrac'/warmUD_freq_.01_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype'-dp.'$dropRate
echo 'Warm Up And Down With Freq .01 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs $nruns -weight-strgy warm_up_down -alt-freq 1 -init-val 0.0 -end-val 1.0 -optimizer $optim -exp-name $exp_fldr'/warmup_.01/'$expname -train-epochs 100 -patience 30 -bn-type $bntype  -dropRate $dropRate -ft-dropRate $dropRate &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'

expname='dataFrac-'$dataFrac'/warmUD_freq_.05_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype'-dp.'$dropRate
echo 'Warm Up And Down With Freq .05 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs $nruns -weight-strgy warm_up_down -alt-freq 5 -init-val 0.0 -end-val 1.0 -optimizer $optim -exp-name $exp_fldr'/warmup_.05/'$expname -train-epochs 100 -patience 30 -bn-type $bntype -dropRate $dropRate -ft-dropRate $dropRate &> run_logs/$expname'.txt' 
tail -n 5 run_logs/$expname'.txt'


expname='dataFrac-'$dataFrac'/warmUD_freq_.2_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype'-dp.'$dropRate
echo 'Warm Up And Down With Freq  .2' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs $nruns -weight-strgy warm_up_down -alt-freq 20 -init-val 0.0 -end-val 1.0 -optimizer $optim -exp-name $exp_fldr'/warmup_.2/'$expname -train-epochs 100 -patience 30 -bn-type $bntype  -dropRate $dropRate -ft-dropRate $dropRate &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


expname='dataFrac-'$dataFrac'/alt_.01_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype'-dp.'$dropRate
echo 'Alternating With Freq .01 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs $nruns -weight-strgy alt -alt-freq 1 -optimizer $optim -exp-name $exp_fldr'/alt_.01/'$expname -train-epochs 100 -patience 30 -bn-type $bntype  -dropRate $dropRate -ft-dropRate $dropRate &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'

expname='dataFrac-'$dataFrac'/alt_.05_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype'-dp.'$dropRate
echo 'Alternating With Freq .05 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs $nruns -weight-strgy alt -alt-freq 5 -optimizer $optim -exp-name $exp_fldr'/alt_.05/'$expname -train-epochs 100 -patience 30 -bn-type $bntype  -dropRate $dropRate -ft-dropRate $dropRate &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


expname='dataFrac-'$dataFrac'/alt_.2_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype'-dp.'$dropRate
echo 'Alternating With Freq .2 ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs $nruns -weight-strgy alt -alt-freq 20 -optimizer $optim -exp-name $exp_fldr'/alt_.2/'$expname -train-epochs 100 -patience 30 -bn-type $bntype  -dropRate $dropRate -ft-dropRate $dropRate &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'



# Including Phase in-and-out

primstart=1
expname='dataFrac-'$dataFrac'/phase-in-'$primstart'_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype'-dp.'$dropRate
echo 'Phase In ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -prim-start $primstart -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs $nruns -weight-strgy phase_in -optimizer $optim -exp-name $exp_fldr'/phasein_.01/'$expname -train-epochs 100 -patience 30 -bn-type $bntype -dropRate $dropRate -ft-dropRate $dropRate &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'

primstart=5
expname='dataFrac-'$dataFrac'/phase-in-'$primstart'_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype'-dp.'$dropRate
echo 'Phase In ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -prim-start $primstart -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs $nruns -weight-strgy phase_in -optimizer $optim -exp-name $exp_fldr'/phasein_.05/'$expname -train-epochs 100 -patience 30 -bn-type $bntype -dropRate $dropRate -ft-dropRate $dropRate &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


primstart=20
expname='dataFrac-'$dataFrac'/phase-in-'$primstart'_ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype'-dp.'$dropRate
echo 'Phase In ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-batch-sz $ftbsz -batch-sz $bsz -prim-datafrac $dataFrac -prim-start $primstart -lr $lr -num-aux-tasks $auxTasks -mode pretrain_w_all -num-runs $nruns -weight-strgy phase_in -optimizer $optim -exp-name $exp_fldr'/phasein_.2/'$expname -train-epochs 100 -patience 30 -bn-type $bntype -dropRate $dropRate -ft-dropRate $dropRate &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'



# Include regular pre-training
expname='dataFrac-'$dataFrac'/regular-pretrain-ntasks='$auxTasks'-lr.'$lr'-optim.'$optim'-ftbsz.'$ftbsz'-bsz.'$bsz'-bntype.'$bntype'-uselast.True-ftlr.'$ftlr'-dp.'$dropRate
echo 'Regular Pretraining ' $expname ' and save file is ' $expname'.txt'
python -u main.py -ft-lr $ftlr -ft-batch-sz $ftbsz -batch-sz 128 -prim-datafrac $dataFrac -lr $lr -num-aux-tasks $auxTasks -mode pretrain -num-runs $nruns -optimizer $optim -exp-name $exp_fldr'/agnostic_pretraining/'$expname -use-last-chkpt -train-epochs 100 -patience 30 -bn-type $bntype -dropRate $dropRate -ft-dropRate $dropRate &> run_logs/$expname'.txt'
tail -n 5 run_logs/$expname'.txt'


echo 'DONE RUNNING - CAN SHUT OFF NOW'
