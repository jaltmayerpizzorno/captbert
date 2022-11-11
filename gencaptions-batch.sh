#!/bin/bash
#SBATCH -c 2            # Number of Cores per Task
#SBATCH --mem=32G       # Requested Memory
#SBATCH -p gpu          # Partition
#SBATCH -G 1            # Number of GPUs
#SBATCH -t 10:00:00     # Job time limit
#SBATCH -o slurm-%j.out # %j = job ID

#conda activate okvqa2
#module load cuda/10.2.89
#cd ~/work/cs646-project/okvqa
which python3

python3 gencaptions.py --load_path rf_model.pth --image_paths data/val2014 --caption_file data/captions-val2014.json
python3 gencaptions.py --load_path rf_model.pth --image_paths data/train2014 --caption_file data/captions-train2014.json
