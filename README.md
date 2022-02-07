This project is a playground meant for testing DVC

1. Download data and create dataset in `data_playground`:

```
cd data_playground
./download_original_MNIST.sh
python convert_mnist.py 0 500 ../data
```

2. Reproduce the experiment:

`dvc repro train`


