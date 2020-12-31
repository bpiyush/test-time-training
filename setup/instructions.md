
### Installing conda

You can run `conda -V` to check if you have `conda` installed. If not, you can follow instructions [here](https://docs.conda.io/projects/conda/en/4.6.1/user-guide/install/index.html).

### Create a conda environment
```
conda create --name ttt python=3.7
conda activate ttt
```

### Install dependencies

#### GPU Machine

Make sure CUDA 10.0 is your default cuda. If your CUDA 10.0 is installed in /usr/local/cuda-10.0, apply

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
```

Install PyTorch and dependencies
```bash
conda install -c pytorch==1.4.0 torchvision=0.5.0 cudatoolkit=10.0
pip install -r requirements.txt
```

#### CPU-only Machine

Install PyTorch and dependencies
```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
pip install -r setup/requirements.txt
```

#### Check installation
```bash
 python -c "import torch; import torchvision; import kornia; import cv2; import numpy"
```
This should run without any errors.

#### Update PYTHONPATH

```bash
export PYTHONPATH=$PWD
```

#### Setup Weights and Biases API Key

If you do not have a [W&B](https://wandb.ai/) account, please create one and obtain your API key from Settings and run:
```bash
export WANDB_API_KEY=<Your Key>
```