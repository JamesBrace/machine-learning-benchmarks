# Utility Benchmarks

This directory holds a handful of useful benchmarks for the components that build up Neural Networks.
This allows for concise benchmarking of individual layers and operations without the hassle of benchmarking an entire
Neural Network. The following benchmarks are included:
* **Convolutional Layer** - Takes in the hyperparameters for the convolutional filter as parameters. In addition, there
is the option to run in regular, transposed, and depthwise modes. 
* **Batch Normalization** - A common operation used before or after activation functions in Neural Networks. Given its wide
use I found it fitting to include it in the repetoire. You can read the paper on Batch Normalization [here](https://arxiv.org/abs/1502.03167).
* **Matrix Multiplication** - The reason for including this is a given! You have the option to perform matrix multiplication
on any size matrix you want! Just feed the size as a parameter.
* **Pooling** - Another widely used layer in CNNs that reduce the feature size in order to speed up training as well as
prevent overfitting. This file includes both Max and Average pooling. In addition to the normal parameters that are fed,
you can also specify the number of times you want to run the layer in CPU mode and get an average of all the run times.
* **Unary Ops** - This includes a handful of useful unary operations such as: log, ReLU, ceiling, sin, tanh and more...
* **Reduction Ops** - This includes a handful of reduction operations such as argMax and mean

### Some notes
* All of these benchmarks have varying input parameters, so you'll need to take a look at the code in order to make sure
you are feed the proper parameters
* Every benchmark has the option to be ran in either 'GPU' or 'CPU' mode which is fed in as a parameter. In order to utilize 
GPU mode with Tensorflow you will need to have CUDA installed with a NVIDIA GPU.. sorry MacBooks ):