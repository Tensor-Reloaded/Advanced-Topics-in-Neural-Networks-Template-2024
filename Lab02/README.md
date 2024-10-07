## Lab 2

***
Lab scripts: 
* How to train a perceptron on MNIST with PyTorch: [perceptron_example_mnist.py](./perceptron_example_mnist.py)
* Training example for downsampled grayscale CIFAR-10: [downsampled_cifar_training.py](./downsampled_cifar_training.py)

***
Homework 2: [Kaggle Competition](https://www.kaggle.com/t/79d63f85ccd848068578901502605679)

***
For self-study (all students):
  * [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
  * [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) (convolution example; convolutions in image processing; convolutions and polynomial multiplication; FFT)
  * https://paperswithcode.com/method/1x1-convolution

Advanced (for students who want to learn more):
* `pin_memory` & `non_blocking=True`:
   * Pinning memory in DataLoaders: https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
   * How does pinned memory actually work: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/ 
   * Also see this discussion: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4

***
References:
 - CIFAR-10 and CIFAR-100: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
 - DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
 - Dataset: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
 - Datasets & DataLoaders example: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
 - TorchVision transforms: https://pytorch.org/vision/stable/transforms.html
 - TorchVision transforms getting started: https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html
 - TorchVision examples: https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
