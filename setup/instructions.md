
### Create a conda environment

conda create --name ttt python=3.7
conda activate ttt

### Install dependencies

Install basic requirements using conda.
```
conda install tqdm
conda install colorama
```

Make sure CUDA 10.0 is your default cuda. If your CUDA 10.0 is installed in /usr/local/cuda-10.0, apply

```
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
```

Install PyTorch, Tensorflow (needed for segmentation) and dependencies
```
conda install pytorch=1.1.0 torchvision cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```