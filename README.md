## Motion Capture Optimization
This is the Final Project Captone of FPT University.

Developed by Anh. Phan Ngoc, Nguyen. Nguyen Trung.

## Environment Setup

(Tested on Ubuntu 22.04; Might work on Windows with some minor modifications)

1.Go to (https://www.anaconda.com/download/) and install the Python 3 version of Anaconda or Miniconda.

2.Open a new terminal and run the following commands to create a new conda environment (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```conda create -n tip22 python=3.8```

3.Activate & enter the new environment you just creared:

```conda activate tip22```

4.Inside the new environment, install CUDA Toolkit, CUDnn with CUDA and the GNU Standard C++ Library :

```
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -c "conda-forge/label/main" cudnn
conda install -c conda-forge libstdcxx-ng
```

5.Install pytorch with CUDA (only tested with the following version, should work with other versions though):

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118``` 

6.Install required library:

```pip install -r requirements.txt```

7.Install our fork of the Fairmotion library, at a location you prefer:
```
git clone https://github.com/jyf588/fairmotion.git
cd fairmotion
pip install -e .
```

## Demo Offline

We have created a special demo, to see it, run the command below::

```python offline.py```