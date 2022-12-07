# CaptBERT: Passage Retrieval With Image Captioning for Outside-Knowledge Visual Question Answering

As my final project for [COMPSCI 646: Information Retrieval](https://groups.cs.umass.edu/zamani/compsci-646-information-retrieval-fall-2022/)
I extended the work in [a paper by Qu, Zamani, Yang, Croft and Learned-Miller](https://github.com/prdwb/okvqa-release) by fine-tuning BERT
to perform dense passage retrieval for OK-VQA based on the question and on image captioning.

## Guide to files
- `gencaptions.py` and `gencaptions-batch.sh` generate the captions; see Captioning below;
- `train.sh` contains the command used for training.  I initially had trouble training on multiple GPUs, and ended up training on a single one;
- `eval.sh` was used to evaluate the checkpoints after training completed (and its automatic evaluation errored out);
- after selecting the best checkpoint, `genreps.sh` was used to generate passage representations for the whole collection. The script requires too much memory to use all at once, but if you use `split` to create files with at most 500,000 lines, the script should require no more than 32GB of memory. This was done on 5 GPUs (I requested 8, but the cluster gave me 5);
- `eval-test.sh` was used to evaluate that checkpoint using the test set, retrieving over the entire collection.

The `logs` directory contains original logs from when (most of) these programs ran.

- `genreps-lxmert.sh`, `eval2-captbert.sh` and `eval2-lxmert.sh` were used to re-run the evaluation after I added paired t-testing.

## Captioning
I selected [ExpansionNet v2](https://github.com/jchenghu/expansionnet_v2) to generate captions,
since at this time it is [one of the newest and best performing caption generators](https://paperswithcode.com/sota/image-captioning-on-coco-captions),
provides code with the paper, and the code seemed relatively easy to reuse.

I modified their [demo.py](https://github.com/jchenghu/ExpansionNet_v2/blob/master/demo.py) to
generate captions using a GPU, and process a few images at a time.
The original `demo.py` took about 4s per image, whereas my modified version takes about .2s.
My version is called [gencaptions.py](gencaptions.py), which I ran using `sbatch` from [gencaptions-batch.sh](gencaptions-batch.sh).

I later merged both training and validation image captions into a single file and changed the dictionary key
from file names to image IDs, to simplify its use.
I modified the scripts to (hopefully) do the same if they were run.
The result is in [data/captions.json](data/captions.json).

## Environment setup
Qu et al.'s code is uses Apex' AMP (automatic mixed precision) support, which is
[incompatible with pytorch 1.10 and later](https://github.com/NVIDIA/apex/issues/1215).
Pytorch 1.9 or earlier isn't readily available for newer Python versions, and neither
are the CUDA libraries, nvcc compiler needed to build Apex, etc., so I tried to keep
things as close as their environment as possible.

These were my steps for setting it up on the Unity cluster utilized at UMass Amherst.
```
module load cuda/10.2.89
module load gcc/8.5.0

conda create --name okvqa2 python=3.8
conda activate okvqa2
conda install faiss-gpu cudatoolkit=10.2 -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install matplotlib

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

I also got it configured for a computer I have at home.
The GPU only has 8GB, so I had to reduce the batch size to 3 (`--per_gpu_train_batch_size=3`), but the GPU (NVIDIA GeForce RTX 2080 SUPER)
is actually a bit faster than those on Unity.
These were the setup steps for my home computer:
```
conda create --name okvqa3 python=3.9
conda activate okvqa3

conda install cudatoolkit-dev=11.1 -c conda-forge
conda install faiss-gpu cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge
pip install --upgrade pip
pip install matplotlib packaging

# avoid wrong CUDA being picked, if installed besides cudatoolkit-dev
unset CUDA_HOME
unset CUDA_PATH

git clone https://github.com/NVIDIA/apex
cd apex

# There is something broken in the current HEAD ()
# related to backwards compatibility with torch.distributed.all_gather_into_tensor
git checkout ce9df7d

# If the GPUs aren't visible where you're compiling, you'll need to set their architectures
# by setting TORCH_CUDA_ARCH_LIST. See e.g. https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# to find out what architecture your GPUs have.  Or use something like "nvidia-smi -q -x | grep product_"
#export TORCH_CUDA_ARCH_LIST="7.0 7.5"

pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd ..

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .
cd ..

conda install -c conda-forge tensorboard
conda install scikit-image
pip install matplotlib datasets pytrec_eval
```

## Acknowledgements
* We thank the authors of:
    * [Qu, Zamani, Yang, Croft and Learned-Miller](https://github.com/prdwb/okvqa-release), upon whose work this one is based;
    * [OK-VQA](https://okvqa.allenai.org/index.html), our dataset;
    * [VQA](https://github.com/GT-Vision-Lab/VQA) and [cocoapi](https://github.com/cocodataset/cocoapi), both used in the software;

See copyright information in LICENSE.
