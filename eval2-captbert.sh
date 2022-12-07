#!/bin/bash
#SBATCH -c 2            # Number of Cores per Task
#SBATCH --mem=32G       # Requested Memory
#SBATCH -p gpu-long     # Partition
#SBATCH -G 1            # Number of GPUs
#SBATCH -t 50:00:00     # Job time limit
#SBATCH -o eval2-captbert-%j.out # %j = job ID

#T=val2014
T=test2014
REPS=`pwd`/../okvqa-reps-captbert/

python -u train_retriever.py \
    --output_dir=`pwd`/../eval-captbert-$T\
    --gen_passage_rep=False \
    --do_train=False\
    --do_eval=True \
    --do_eval_pairs=False \
    --val_data_sub_type=$T \
    --collection_reps_path=$REPS \
    --retrieve_checkpoint=`pwd`/data/checkpoint-captbert \
    --baseline_path=`pwd`/data/checkpoint-lxmert \
    --per_gpu_eval_batch_size=6 \
    --fp16=True \
    --load_small=False \
    --num_workers=1 \
    --query_encoder_type=capt-bert\
    --query_model_name_or_path="bert-base-uncased" \
    --proj_size=768 \
    --neg_type=other_pos+all_neg
