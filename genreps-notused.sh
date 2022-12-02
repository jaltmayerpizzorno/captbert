#!/bin/bash

OUT=`pwd`/../okvqa-reps-captbert
mkdir $OUT

for F in `pwd`/data/all_blocks_split/*; do
    echo "$N"
    N=`basename $F`
    python -u train_retriever.py \
        --gen_passage_rep=True \
        --gen_passage_rep_input=$F \
        --gen_passage_rep_output=$OUT/$N \
        --retrieve_checkpoint=`pwd`/data/checkpoint-captbert \
        --do_train=False \
        --do_eval=False \
        --do_eval_pairs=False \
        --per_gpu_eval_batch_size=225 \
        --fp16=True \
        --load_small=False \
        --num_workers=1 \
        --query_encoder_type=capt-bert\
        --query_model_name_or_path="bert-base-uncased" \
        --proj_size=768 \
        --neg_type=other_pos+all_neg
done
