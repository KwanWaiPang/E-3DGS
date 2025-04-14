<div align="center">
<h1>测试 （3DV 2025）E-3DGS: Event-based Novel View Rendering of Large-scale Scenes Using 3D Gaussian Splatting</h1>
</div>

# 配置测试
```bash
git clone https://github.com/KwanWaiPang/E-3DGS.git --recursive

# rm -rf .git


conda env create --yes --file environment_cuda12.2.yml #注意A100需要采用这个
#conda env create --yes --file environment_cuda11.6.yml #为cuda11.7采用的版本
conda activate E-3DGS
# conda remove --name E-3DGS --all

conda install pytorch3d -c pytorch3d
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
# pip install opencv-python pandas piq scipy numba tensorboard matplotlib lpips
pip install pandas piq scipy matplotlib lpips

```