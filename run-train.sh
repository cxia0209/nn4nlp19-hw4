#!/bin/bash
export NUM_CORES=1
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES MKL_DYNAMIC=false
python code/main.py --ac_share_emb True --batch_size 1 --seed $1 --optim adadelta --lr 1.0 --rnn_type $2 --pos_sp True --output_pred --covered-test --epochs 10 --patience 10 -d $3 
