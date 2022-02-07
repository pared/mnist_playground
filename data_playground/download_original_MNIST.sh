#!/bin/bash

dvc get-url http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz ./raw/MNIST/train-images-idx3-ubyte.gz
dvc get-url http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz ./raw/MNIST/train-labels-idx1-ubyte.gz
dvc get-url http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz  ./raw/MNIST/t10k-images-idx3-ubyte.gz
dvc get-url http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz  ./raw/MNIST/t10k-labels-idx1-ubyte.gz

