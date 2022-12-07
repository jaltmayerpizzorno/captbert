#!/bin/bash
#SBATCH -c 8            # Number of Cores per Task
#SBATCH --mem=32G       # Requested Memory
#SBATCH -p gpu-long     # Partition
#SBATCH -G 8            # Number of GPUs
#SBATCH -t 50:00:00     # Job time limit
#SBATCH -o genreps-%j.out # %j = job ID

module load cuda/10.2.89

OUT=`pwd`/../okvqa-reps-captbert
mkdir $OUT

for F in `pwd`/data/all_blocks_split/*; do
    echo "$N"
    N=`basename $F`
    # don't use torch.distributed.launch for inference
    python -u train_retriever.py \
        --gen_passage_rep=True \
        --gen_passage_rep_input=$F \
        --gen_passage_rep_output=$OUT/$N \
        --retrieve_checkpoint=`pwd`/data/checkpoint-captbert \
        --do_train=False \
        --do_eval=False \
        --do_eval_pairs=False \
        --per_gpu_eval_batch_size=300 \
        --fp16=True \
        --load_small=False \
        --num_workers=8 \
        --query_encoder_type=captbert\
        --query_model_name_or_path="bert-base-uncased" \
        --proj_size=768 \
        --neg_type=other_pos+all_neg
done
