#!/bin/bash
#SBATCH -c 8            # Number of Cores per Task
#SBATCH --mem=64G       # Requested Memory
#SBATCH -p gpu-long     # Partition
#SBATCH -G 8            # Number of GPUs
#SBATCH -t 50:00:00     # Job time limit
#SBATCH -o eval-test-%j.out # %j = job ID

module load cuda/10.2.89

REPS=`pwd`/../okvqa-reps-captbert/

# don't use torch.distributed.launch for inference
python -u train_retriever.py \
    --gen_passage_rep=False \
    --do_train=False \
    --do_eval=True \
    --do_eval_pairs=False \
    --val_data_sub_type=test2014 \
    --collection_reps_path=$REPS \
    --retrieve_checkpoint=`pwd`/data/checkpoint-captbert \
    --per_gpu_eval_batch_size=10 \
    --fp16=True \
    --load_small=False \
    --num_workers=8 \
    --query_encoder_type=captbert \
    --query_model_name_or_path="bert-base-uncased" \
    --proj_size=768 \
    --neg_type=other_pos+all_neg
