#!/bin/bash
export NUM_CORES=1
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES MKL_DYNAMIC=false
python code/main.py --ac_share_emb True --batch_size 1 --seed 9527 --optim adadelta --lr 1.0 --rnn_type lstm --model AdvTransducer --pos_sp True --output_pred --covered-test --epochs 60 --patience 10 --roll_in_k 12 --oversample 5 --lambda-q 0.01 --sampler OverBalancedRandomSampler -d $1 
