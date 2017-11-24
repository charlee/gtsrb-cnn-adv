import os
import tarfile
import pickle
import numpy as np
from .base import DatasetProvider
import cnn_gtsrb.settings as settings


class Cifar10Provider(DatasetProvider):
    IMAGE_SIZE = 32
    CLASSES = 10
    CHANNELS = 3
    URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    DATA_DIR = settings.DATA_TEMP_CIFAR10


    def init(self):
        self.data_dir = os.path.join(self.DATA_DIR, 'cifar10_data')

        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)

        if not os.path.isfile(os.path.join(self.data_dir, 'training.npy')):

            # Download tar.gz file if not downloaded yet
            rawdata_dir = os.path.join(self.DATA_DIR, 'rawdata')
            filepath = os.path.join(rawdata_dir, self.URL.split('/')[-1])
            if not os.path.isfile(filepath):
                self.download(self.URL, filepath)

            tar = tarfile.open(filepath, 'r:gz')
            tar.extractall(rawdata_dir)
            tar.close()

            training_set = []
            for file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
                filepath = os.path.join(rawdata_dir, 'cifar-10-batches-py', file)
                with open(filepath, 'rb') as f:
                    d = pickle.load(f, encoding='bytes')
                    images = d[b'data']
                    labels = d[b'labels']

                    # Change images layout from (RRRR...GGGG...BBBB...) to (RGBRGBRGB...)
                    images = np.reshape(images, [-1, self.CHANNELS, self.IMAGE_SIZE, self.IMAGE_SIZE])
                    images = np.moveaxis(images, 1, 3)
                    images = np.reshape(images, [-1, self.CHANNELS * self.IMAGE_SIZE * self.IMAGE_SIZE])
                    labels = np.expand_dims(labels, 1)
                    data = np.concatenate([images, labels], axis=1)

                    training_set.append(data)
        
            training_set = np.concatenate(training_set, axis=0)
            training_set.dump(os.path.join(self.data_dir, 'training.npy'))

            filepath = os.path.join(rawdata_dir, 'cifar-10-batches-py', 'test_batch')
            with open(filepath, 'rb') as f:
                d = pickle.load(f, encoding='bytes')
                images = d[b'data']
                labels = d[b'labels']

                images = np.reshape(images, [-1, self.CHANNELS, self.IMAGE_SIZE, self.IMAGE_SIZE])
                images = np.moveaxis(images, 1, 3)
                images = np.reshape(images, [-1, self.CHANNELS * self.IMAGE_SIZE * self.IMAGE_SIZE])
                labels = np.expand_dims(labels, 1)

                test_data = np.concatenate([images, labels], axis=1)

            test_data.dump(os.path.join(self.data_dir, 'test.npy'))
            