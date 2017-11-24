import os
import re
import numpy as np
from cnn_gtsrb import settings
from .mnist import MnistProvider

class MnistBgProvider(MnistProvider):
    name = 'mnist_bg'
    URL = 'http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip'
    IMAGE_SIZE = 28
    CLASSES = 10
    CHANNELS = 1
    DATA_DIR = settings.DATA_TEMP_MNIST_BG

    def init(self):

        self.data_dir = os.path.join(self.DATA_DIR, 'mnist_bg_data')
        rawdata_dir = os.path.join(self.DATA_DIR, 'rawdata')

        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)

        if not os.path.isfile(os.path.join(self.data_dir, 'training.npy')):
            # Download zip file if not downloaded yet
            filepath = os.path.join(rawdata_dir, self.URL.split('/')[-1])
            if not os.path.isfile(filepath):
                self.download_and_unzip(self.URL, filepath, os.path.join(rawdata_dir, 'unzipped'))

            # parse training data
            training_data = self.read_data(os.path.join(rawdata_dir, 'unzipped', 'mnist_background_images_test.amat'))
            test_data = self.read_data(os.path.join(rawdata_dir, 'unzipped', 'mnist_background_images_train.amat'))

            training_data.dump(os.path.join(self.data_dir, 'training.npy'))
            test_data.dump(os.path.join(self.data_dir, 'test.npy'))

    def read_data(self, filepath):
        images = []
        labels = []
        lines = open(filepath, 'r').readlines()
        for line in lines:
            data = [float(s) for s in re.split(r'\s+', line) if s]
            images.append(data[:-1])
            labels.append(data[-1])

        images = np.array(images) * 255
        images = images.astype(np.uint8)

        # transpose image matrix
        images = np.reshape(images, [-1, self.IMAGE_SIZE, self.IMAGE_SIZE])
        images = np.moveaxis(images, 1, 2)
        images = np.reshape(images, [-1, self.IMAGE_SIZE * self.IMAGE_SIZE])

        labels = np.array(labels).astype(np.uint8)
        labels = np.expand_dims(labels, 1)

        return np.concatenate([images, labels], axis=1)