
1. Use `download_original_MNIST.sh` from within this dir:

   `./download_original_MNIST.sh`


2. Use `convert_mnist.py` to generate partial MNIST:

   `python convert_mnist.py 0 100 dirname`

   will write first 100 images of each label for both test and train dataset into `dirname/test` and `dirname/train` 
