#!/bin/bash
#SBATCH -c 6            # Number of Cores per Task
#SBATCH --mem=32G       # Requested Memory
#SBATCH -p gpu-long     # Partition
#SBATCH -G 6            # Number of GPUs
#SBATCH -t 30:00:00     # Job time limit
#SBATCH -o genreps-lxmert-%j.out # %j = job ID

module load cuda/10.2.89

OUT=`pwd`/../okvqa-reps-lxmert
mkdir $OUT

for F in `pwd`/data/all_blocks_split/*; do
    N=`basename $F`
    if [[ -f "$OUT/$N" ]]; then
        continue
    fi
    echo "$N"
    # don't use torch.distributed.launch for inference
    python -u train_retriever.py \
        --gen_passage_rep=True \
        --gen_passage_rep_input=$F \
        --gen_passage_rep_output=$OUT/$N \
        --retrieve_checkpoint=`pwd`/data/checkpoint-lxmert \
        --do_train=False \
        --do_eval=False \
        --do_eval_pairs=False \
        --per_gpu_eval_batch_size=250 \
        --fp16=True \
        --load_small=False \
        --num_workers=6 \
        --query_encoder_type=lxmert \
        --query_model_name_or_path="unc-nlp/lxmert-base-uncased" \
        --lxmert_rep_type="{\"pooled_output\":\"none\"}" \
        --proj_size=768 \
        --neg_type=other_pos+all_neg
done
