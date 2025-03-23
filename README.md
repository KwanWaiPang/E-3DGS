This repository contains the official code for the research paper titled "Event 3D Gaussian Splatting: Event-based Novel View Rendering of Large-scale Scenes using 3D Gaussians". The code uses [3D Gaussian Splatting (3DGS)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) as it's base. Please refer to `README_3DGS.md` in this repository for the readme provided with the original repository of 3DGS.

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

# Installation

You can setup the environment for this code base by running the following bash commands:

```shell
conda env create --yes --file environment.yml
conda activate splat
conda install pytorch3d -c pytorch3d
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install opencv-python pandas piq scipy numba tensorboard matplotlib lpips
```

The compilation of submodules is dependant on the debian version. Hence to have a working environment for slurm, submit the slurm script named `setup_env.sh` after adjusing paths as per your conda installation. The script will install the necesssary dependancies into `splat` environment.

# Training

Firstly download the sample dataset [link](https://drive.google.com/file/d/1AfWS1Pp0Sl_3fRgUuOQ3JLNX86pq3Tn_/view?usp=sharing) and extract it into a directory. The full model can then be trained by running the following command:

```shell
python train.py -s /path/to/data/dir/shot_009 -m /path/to/model/output/dir --pose_lr 0.001 --sh_degree 1
```

# Inference

To render the novel views from the test set, run the following command:

```shell
python render.py -s /path/to/data/dir/shot_009 -m /path/to/model/output/dir --skip_train
```
Remove the `--skip_train` flag if training views are also required to be rendered.

The rendered images can then be found in the model output directory.

# Full list of experiments for the paper

The script used to run the entire list of experiments listed in the paper can be found with the name of `run_experiment.sh`. It can also be used for general understanding of the experimentation setting.

# Citation

Please cite our work if you use the code.

```
@article{zahid2025e3dgs,
  title={E-3DGS: Event-based Novel View Rendering of Large-scale Scenes Using 3D Gaussian Splatting},
  author={Zahid, Sohaib and Rudnev, Viktor and Ilg, Eddy and Golyanik, Vladislav},
  journal={3DV},
  year={2025}
}
```
