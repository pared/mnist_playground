import gzip
import os
import shutil
import sys

import idx2numpy
import numpy as np
from PIL import Image


def load(path):
    with gzip.open(path, "rb") as fd:
        return idx2numpy.convert_from_file(fd)


def write_images(data, labels, path, from_index, to_index):
    assert len(data) == len(labels)
    indexes = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for i, image in enumerate(data):

        label = labels[i]
        label_path = os.path.join(path, str(label))
        index = indexes[label]

        if index >= from_index and index < to_index:
            os.makedirs(label_path, exist_ok=True)
            img_path = os.path.join(label_path, str(index) + ".jpg")
            im = Image.fromarray(image)
            im.save(img_path)

        if all(val >= to_index for val in indexes.values()):
            break
        indexes[label] += 1


if __name__ == "__main__":
    try:
        start, end, target_path = sys.argv[1:]
    except ValueError as e:
        print(f"Expecting 'stat_index', 'end_index' and 'path' to write into!")
        sys.exit(1)
    start = int(start)
    end = int(end)
    shutil.rmtree(target_path, ignore_errors=True)
    print(f"Writing images from '{start}' to '{end}' indexes into '{target_path}'.")

    test_data = load(os.path.join("raw", "MNIST", "t10k-images-idx3-ubyte.gz"))
    train_data = load(os.path.join("raw", "MNIST", "train-images-idx3-ubyte.gz"))
    test_labels = load(os.path.join("raw", "MNIST", "t10k-labels-idx1-ubyte.gz"))
    train_labels = load(os.path.join("raw", "MNIST", "train-labels-idx1-ubyte.gz"))

    write_images(
        train_data, train_labels, os.path.join(target_path, "train"), start, end
    )
    write_images(test_data, test_labels, os.path.join(target_path, "test"), start, end)
