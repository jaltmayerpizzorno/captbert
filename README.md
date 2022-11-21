# Passage Retrieval With Automatically Generated Captions for Outside-Knowledge Visual Question Answering

(work-in-progress)

As my final project for [COMPSCI 646: Information Retrieval](https://groups.cs.umass.edu/zamani/compsci-646-information-retrieval-fall-2022/)
I am extending the work in [a paper by Qu, Zamani, Yang, Croft and Learned-Miller](https://github.com/prdwb/okvqa-release) by fine-tuning BERT
to perform dense passage retrieval for OK-VQA based on the question and on an automatically generated caption.

## Environment setup
Qu et al.'s code is based on Python 3.8, so to minimize issues I thought I'd use that as well.
However, current Python is now 3.11 and various packages have moved on, so it's not all a walk in the park.

These steps are for setting it up on the Unity cluster.
```
module load cuda/10.2.89
module load gcc/8.5.0

conda create --name okvqa2 python=3.8
conda activate okvqa2
conda install faiss-gpu cudatoolkit=10.2 -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts

# --- apex (distributed machine learning package)
git clone https://github.com/NVIDIA/apex
cd apex

# After ce9df7d a change was made that is incompatible with Python 3.8
git checkout ce9df7d

# If the GPUs aren't visible where you're compiling, you'll need to set their architectures
# by setting TORCH_CUDA_ARCH_LIST. See e.g. https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# to find out what architecture your GPUs have.  Or use something like "nvidia-smi -q -x | grep product_"
export TORCH_CUDA_ARCH_LIST="7.0 7.5"

# The compilation uses so much CPU that it may be killed (because of system policies);
# run it in the cluster's CPU partition instead.
srun --pty -p cpu -c 10 pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
# -- apex

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
cd ..

conda install -c conda-forge tensorboard
pip install datasets
pip install pytrec_eval
conda install scikit-image
```

## Generating Captions

I selected [ExpansionNet v2](https://github.com/jchenghu/expansionnet_v2) to generate captions,
since at this time it is [one of the newest and best performing caption generators](https://paperswithcode.com/sota/image-captioning-on-coco-captions),
provides code with the paper, and the code seemed relatively easy to reuse.

I modified their [demo.py](https://github.com/jchenghu/ExpansionNet_v2/blob/master/demo.py) to
generate captions using a GPU, and process a few images at a time.
The original `demo.py` took about 4s per image, whereas my modified version takes about .2s.
My version is called [gencaptions.py](gencaptions.py), which I ran using `sbatch` from [gencaptions-batch.sh](gencaptions-batch.sh).

The generated captions are in [data/captions-train2014.json](data/captions-train2014.json) and [data/captions-val2014.json](data/captions-val2014.json).


---
## The items below are still original notes from Qu et al.'s OK-VQA

### Training, and evaluating on the validation set with the small validation collection
```
python -u -m torch.distributed.launch --nproc_per_node 4 train_retriever.py \
    --input_file=DATA_DIR/data/{}_pairs_cap_combine_sum.txt \
    --image_features_path=DATA_DIR/okvqa.datasets \
    --output_dir=OUTPUT_DIR \
    --ann_file=DATA_DIR/mscoco_val2014_annotations.json \
    --ques_file=DATA_DIR/OpenEnded_mscoco_val2014_questions.json \
    --passage_id_to_line_id_file=DATA_DIR/passage_id_to_line_id.json \
    --all_blocks_file=DATA_DIR/all_blocks.txt \
    --gen_passage_rep=False \
    --gen_passage_rep_input=DATA_DIR/val2014_blocks_cap_combine_sum.txt \
    --cache_dir=HUGGINGFACE_CACHE_DIR (optional) \
    --gen_passage_rep_output="" \
    --retrieve_checkpoint="" \
    --collection_reps_path="" \
    --val_data_sub_type=val2014 \
    --do_train=True \
    --do_eval=True \
    --do_eval_pairs=True \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=10 \
    --learning_rate=1e-5 \
    --num_train_epochs=2.0 \
    --logging_steps=5 \
    --save_steps=5000 \
    --eval_all_checkpoints=True \
    --overwrite_output_dir=False \
    --fp16=True \
    --load_small=False \
    --num_workers=4 \
    --query_encoder_type=lxmert \
    --query_model_name_or_path="unc-nlp/lxmert-base-uncased" \
    --lxmert_rep_type="{\"pooled_output\":\"none\"}" \
    --proj_size=768 \
    --neg_type=other_pos+all_neg
```

If experiencing OOM during evaluation, the evaluation process can be restarted with the same command but set `--do_train=False` and `--overwrite_output_dir=True`.

### Generating representations for all passages in the collection
```
python -u train_retriever.py \
    --input_file=DATA_DIR/data/{}_pairs_cap_combine_sum.txt \
    --image_features_path=DATA_DIR/okvqa.datasets \
    --output_dir=OUTPUT_DIR \
    --ann_file=DATA_DIR/mscoco_val2014_annotations.json \
    --ques_file=DATA_DIR/OpenEnded_mscoco_val2014_questions.json \
    --passage_id_to_line_id_file=DATA_DIR/passage_id_to_line_id.json \
    --all_blocks_file=DATA_DIR/all_blocks.txt \
    --gen_passage_rep=True \
    --gen_passage_rep_input=DATA_DIR/all_blocks.txt (or a split of this file) \
    --cache_dir=HUGGINGFACE_CACHE_DIR (optional) \
    --gen_passage_rep_output=OUTPUT_DIR_OF_PASSAGE_REPS \
    --retrieve_checkpoint=DIR_TO_YOUR_BEST_CHECKPOINT \
    --collection_reps_path="" \
    --val_data_sub_type=test2014 (doesn't matter here) \
    --do_train=False \
    --do_eval=False \
    --do_eval_pairs=False \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=300 \
    --learning_rate=1e-5 \
    --num_train_epochs=2.0 \
    --logging_steps=5 \
    --save_steps=5000 \
    --eval_all_checkpoints=True \
    --overwrite_output_dir=True \
    --fp16=True \
    --load_small=False \
    --num_workers=8 \
    --query_encoder_type=lxmert \
    --query_model_name_or_path=unc-nlp/lxmert-base-uncased \
    --lxmert_rep_type="{\"pooled_output\":\"none\"}" \
    --proj_size=768 \
    --neg_type=other_pos+all_neg
```

### Evaluating on the test set with the whole passage collection
```
python -u train_retriever.py \
    --input_file=DATA_DIR/data/{}_pairs_cap_combine_sum.txt \
    --image_features_path=DATA_DIR/okvqa.datasets \
    --output_dir=OUTPUT_DIR \
    --ann_file=DATA_DIR/mscoco_val2014_annotations.json \
    --ques_file=DATA_DIR/OpenEnded_mscoco_val2014_questions.json \
    --passage_id_to_line_id_file=DATA_DIR/passage_id_to_line_id.json \
    --all_blocks_file=DATA_DIR/all_blocks.txt \
    --gen_passage_rep=False \
    --gen_passage_rep_input=DATA_DIR/val2014_blocks_cap_combine_sum.txt \
    --cache_dir=HUGGINGFACE_CACHE_DIR (optional) \
    --gen_passage_rep_output="" \
    --retrieve_checkpoint=DIR_TO_YOUR_BEST_CHECKPOINT \
    --collection_reps_path=DIR_OF_PASSAGE_REPS \
    --val_data_sub_type=test2014 \
    --do_train=False \
    --do_eval=True \
    --do_eval_pairs=False \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=10 \
    --learning_rate=1e-5 \
    --num_train_epochs=2.0 \
    --logging_steps=5 \
    --save_steps=5000 \
    --eval_all_checkpoints=True \
    --overwrite_output_dir=True \
    --fp16=True \
    --load_small=False \
    --num_workers=1 \
    --query_encoder_type=lxmert \
    --query_model_name_or_path="unc-nlp/lxmert-base-uncased" \
    --lxmert_rep_type="{\"pooled_output\":\"none\"}" \
    --proj_size=768 \
    --neg_type=other_pos+all_neg
```



To generate image features, we need `opencv`, which doesn't work with python 3.8 at the moment. So image features are generated with an environment with python 3.7 (no need to do this if you have downloaded the features we extracted linked above). Use command `conda install -c conda-forge opencv` to install `opencv`.

## Acknowledgement
* Our training data is based on [OK-VQA](https://okvqa.allenai.org/index.html). We thank the OK-VQA authors for creating and releasing this useful resource.  
* `vqa_tools.py` is built on [VQA](https://github.com/GT-Vision-Lab/VQA). We thank the VQA authors for releasing their code.  
* `coco_tools.py` is built on [cocoapi](https://github.com/cocodataset/cocoapi). We thank the cocoapi authors for releasing their code.  

See copyright information in LICENSE.
