#!/bin/bash
#SBATCH -c 2            # Number of Cores per Task
#SBATCH --mem=32G       # Requested Memory
#SBATCH -p gpu-long     # Partition
#SBATCH -G 1            # Number of GPUs
#SBATCH -t 50:00:00     # Job time limit
#SBATCH -o slurm-%j.out # %j = job ID

python train_retriever.py \
    --output_dir=`pwd`/../okvqa_output-2\
    --gen_passage_rep=False \
    --val_data_sub_type=val2014 \
    --do_train=False\
    --do_eval=True \
    --do_eval_pairs=True \
    --per_gpu_train_batch_size=3 \
    --per_gpu_eval_batch_size=6 \
    --learning_rate=1e-5 \
    --num_train_epochs=2.0 \
    --logging_steps=5 \
    --save_steps=5000 \
    --fp16=True \
    --load_small=False \
    --num_workers=1 \
    --query_encoder_type=captbert\
    --query_model_name_or_path="bert-base-uncased" \
    --proj_size=768 \
    --neg_type=other_pos+all_neg
