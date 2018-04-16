# Machine Learning Benchmarks: Server vs Browser

The purpose of this repository is to start creating a collection of machine learning algorithms in both their server-side
and browser-side implementations in order to gauge their relative performance. In the past few years there has been rapid
development in browser-side deep learning libraries such as ConvNet.js, TensorFire, and Deeplearn.js (now Tensorflow.js).
These libraries have allowed developers to tap into high-performance Machine Learning with minimial overhead, making Machine
Learning more accessible than ever. Each model in this repository contains both it's server-side (written with TensorFlow) 
as well as it's browser-side (written with Deeplearn) implementations and can be run with little-to-no setup!

This repository has begun as an honours computer science project at McGill University, but I hope to see it grow over the
next fews years as more state-of-the-art algorithms (and libraries) come out.

## What's Included
In each folder, there sits an out-of-the-box implementation of various machine learning algorithm. 

Currently there are the following benchmarks:
* **MNIST Convolutional Network** - this is the typical CNN you will find in most 'Get Started' tutorials. Nothing too
fancy! Yet given it's relevancy I thought it fitting to be included. 
* **SqueezeNet** - this latest state-of-the-art network records AlexNet-like performance, but used only a fraction of
the parameters. Because of this, it is a suitable target for devices that need to fetch models over a server, or limited 
computational resources. Special thanks to @vonclites for his TensorFlow implementation. The DeepLearn version is a line-by-line
translation of his code. 
* **Utility Functions** - this includes a handful of utility functions needed to construct Neural Networks. This includes:
ReLU activation (and all of its counterparts), the Conv2D layer function, and various reduction and unary function calls. 


## What's Required
As of now, the only requirements needed to run each successfully is:

* [Python 3](https://www.python.org/downloads/) - needed to run the server-side implementation of the algorithms
* [TensorFlow](https://www.tensorflow.org/) - the Deep Learning library used to construct all the models for the server-side code
* [TypeScript](https://www.typescriptlang.org/) - all the browser-side code (except for MNIST) is written in TypeScript
* [Browserify](http://browserify.org/) - is need to compile some of the TypeScript modules
* and of course a modern browser (we suggest Chrome or FireFox) :)

Each directory has a README with more specific instructions to run the code.

## What's To Come
* A link to the final report highlighting and discussing the outcomes of the benchmarks as well as the benchmarking methodology
used to rigourously analyse the data.
* Implementation of [DenseNet](https://arxiv.org/abs/1608.06993)
* Implementation of [MobileNet](https://arxiv.org/abs/1704.04861)
* Implementation of [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
* Implementation of [Inception V3](https://arxiv.org/abs/1512.00567)
* Continuous iterations of improvement as I become more familar with libaries

### Contributors
* James Brace (Me) - Honours Computer Science student at McGill University. You can contact me at [jamesbrace@mail.mcgill.ca]()

### Special Thanks
* @vonclites for his SqueezeNet implementation
* The Google Brain team for making all their Deep Learning libraries open-source and accessible!