## Lab 2

***
Lab Notebooks:
* [Augmentations.ipynb](./Augmentations.ipynb)

Lab scripts: 
* How to train a perceptron on MNIST with PyTorch: [perceptron_example_mnist.py](./perceptron_example_mnist.py)

See [CIFAR10](./CIFAR10) for a training and inference example on CIFAR-10.

***
Homework 2: [Kaggle Competition](https://www.kaggle.com/t/79d63f85ccd848068578901502605679)

***
For self-study (all students):
  * [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
  * [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) (convolution example; convolutions in image processing; convolutions and polynomial multiplication; FFT)
  * https://paperswithcode.com/method/1x1-convolution
  * TorchVision transforms getting started: https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html
  * TorchVision examples: https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py


Advanced (for students who want to learn more):
* `pin_memory` & `non_blocking=True`:
   * Pinning memory in DataLoaders: https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
   * How does pinned memory actually work: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/ 
   * Also see this discussion: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4
* Data Augmentation Techniques:
  * [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)
  * [Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899)
  * [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
  * RandAugment in torchvision: https://pytorch.org/vision/main/generated/torchvision.transforms.v2.RandAugment.html
  * How to use CutMix and MixUp: https://pytorch.org/vision/main/auto_examples/transforms/plot_cutmix_mixup.html


***
References:
 - DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
 - Dataset: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
 - Datasets & DataLoaders example: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
 - CIFAR-10 and CIFAR-100: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
 - CIFAR-10 training example: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
