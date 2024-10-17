![image_clipdrop-enhance](https://github.com/Tensor-Reloaded/Advanced-Topics-in-Neural-Networks-Template-2023/assets/8055539/5965f7aa-34ad-4899-b2af-be3cc084cb96)

# [Advanced-Topics-in-Neural-Networks-Template-2024](https://sites.google.com/view/rbenchea/advanced-chapters-of-neural-networks)

Repository for the Advanced Topics in Neural Networks laboratory, "Alexandru Ioan Cuza" University, Faculty of Computer Science, Master degree.

## Environment setup

Google Colab: PyTorch, Pandas, Numpy, Tensorboard, and Matplotlib are already available. Wandb can be easily installed using `pip install wandb`. 

Local instalation: 
1. Create a Python environment (using conda or venv). We recommend installing conda from [Miniforge](https://github.com/conda-forge/miniforge).
```
# Create the environment
conda create -n 312 -c conda-forge python=3.12
# activate the environment
conda activate 312
# Run this to use conda-forge as your highest priority channel (not needed if you installed conda from Miniforge)
conda config --add channels conda-forge
```
2. Install PyTorch 2.4.1+ from [pytorch.org](https://pytorch.org/get-started/locally/) using `conda` or `pip`, depending on your environment. 
    * Choose the Stable Release, choose your OS, select Conda or Pip and your compute platform. For Linux and Windows, CUDA or CPU builds are available, while for Mac, only builds with CPU and MPS acceleration.
    * Example CPU: ```conda install pytorch torchvision torchaudio cpuonly -c pytorch```.
3. Install Tensorboard and W&B
    * `conda install -c conda-forge tensorboard wandb`
4. Install Matplotlib.
     * `conda install conda-forge::matplotlib`

## Recommended resources:

- Linear algebra:
   * [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (linear transformations; matrix multiplication)
   * [Essence of calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) (derivatives; chain rule)
- Backpropagation:
   * [Neural Networks (chapter 1 - chapter 4)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (animated introduction to neural networks and backpropagation)
- Convolutions:
   * [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) (convolution example; convolutions in image processing; convolutions and polynomial multiplication; FFT)
- Transformers:
   * [Neural Networks (chapter 5 - chapter 7)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (GPT; visual explanation of attention; LLMs)
- Also see [Resources.md](https://github.com/Tensor-Reloaded/Advanced-Topics-in-Neural-Networks-Template-2024/blob/main/Resources.md).
  
## Table of contents

* [Lab01](./Lab01): Tensor Operations (Homework 1: Multi Layer Perceptron + Backpropagation)
* [Lab02](./Lab02): Convolutions, DataLoaders, Datasets, Data Augmentation techniques (Homework 2: Kaggle competition on CIFAR-100 with VGG-16)
* [Lab03](./Lab03): ResNets (Homework 3: Implement a complete training pipeline with PyTorch)

## [2023 archive](https://github.com/Tensor-Reloaded/Advanced-Topics-in-Neural-Networks-Template-2023)
