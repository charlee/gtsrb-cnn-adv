import os
import gzip
import numpy as np
from .base import DatasetProvider
import cnn_gtsrb.settings as settings

class MnistProvider(DatasetProvider):
    IMAGE_SIZE = 28
    CLASSES = 10

    URL_BASE = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    DATA_DIR = settings.DATA_TEMP_MNIST

    def init(self):
        self.data_dir = os.path.join(self.DATA_DIR, 'mnist_data')

        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)

            rawdata_dir = os.path.join(self.DATA_DIR, 'rawdata')

            # Download MNIST
            for f in (self.TRAIN_IMAGES, self.TRAIN_LABELS, self.TEST_IMAGES, self.TEST_LABELS):
                filepath = os.path.join(rawdata_dir, f)
                if not os.path.isfile(filepath):
                    self.download(self.URL_BASE + f, filepath)

            training_images = self.read_image_file(os.path.join(rawdata_dir, self.TRAIN_IMAGES))
            test_images = self.read_image_file(os.path.join(rawdata_dir, self.TEST_IMAGES))
            training_labels = self.read_label_file(os.path.join(rawdata_dir, self.TRAIN_LABELS))
            test_labels = self.read_label_file(os.path.join(rawdata_dir, self.TEST_LABELS))

            training_set = self.merge_images_labels(training_images, training_labels)
            test_set = self.merge_images_labels(test_images, test_labels)

            training_set.dump(os.path.join(self.data_dir, 'training.npy'))
            test_set.dump(os.path.join(self.data_dir, 'test.npy'))


    def merge_images_labels(self, images, labels):
        return np.concatenate((
            np.reshape(images, [-1, self.IMAGE_SIZE * self.IMAGE_SIZE]),
            np.reshape(labels, [-1, 1]),
        ), axis=1)



    def _as_int32(self, arr):
        result = np.fromstring(arr[:4].tostring(), dtype='>i4')
        return result[0]

    def read_image_file(self, filepath):
        """Read MNIST image file and return a ndarray of all image files."""
        with gzip.open(filepath, 'rb') as f:
            content = f.read()
            data = np.fromstring(content, dtype=np.uint8)
            assert(self._as_int32(data[0:4]) == 0x00000803)
            count = self._as_int32(data[4:8])
            height = self._as_int32(data[8:12])
            width = self._as_int32(data[12:16])

            images = data[16:]
            assert(images.shape[0] == count * height * width)

            images = np.reshape(images, [count, height, width])

            return images

    def read_label_file(self, filepath):
        """Read MNIST label file and return a ndarray of all label files."""
        with gzip.open(filepath, 'rb') as f:
            content = f.read()
            data = np.fromstring(content, dtype=np.uint8)
            assert(self._as_int32(data[0:4]) == 0x00000801)
            count = self._as_int32(data[4:8])

            labels = data[8:]
            assert(labels.shape[0] == count)

            return labels