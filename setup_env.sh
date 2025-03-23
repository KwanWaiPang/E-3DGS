#!/bin/bash
#SBATCH -p gpu22
#SBATCH --mem=16G
#SBATCH --signal=B:SIGTERM@120
#SBATCH -o ./output/setup.out
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task 2
#SBATCH --gres gpu:1
 
eval "$(/CT/EventSLAM/work/miniconda3/bin/conda shell.bash hook)"

conda remove --yes -n splat --all
conda env create --yes --file environment.yml
conda activate splat
conda install pytorch3d -c pytorch3d
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install opencv-python pandas piq scipy numba tensorboard matplotlib lpips
