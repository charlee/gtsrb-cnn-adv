import os
import gzip
import numpy as np
from .base import DatasetProvider
import cnn_gtsrb.settings as settings


class Cifar10Provider(DatasetProvider):
    IMAGE_SIZE = 32
    CLASSES = 10
    URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    DATA_DIR = settings.DATA_TEMP_CIFAR10


    def init(self):
        self.data_dir = os.path.join(self.DATA_DIR, 'cifar10_data')

        rawdata_dir = os.path.join(self.DATA_DIR, 'rawdata')
        filepath = os.path.join(rawdata_dir, self.URL.split('/')[-1])
        if not.os.path.isfile(filepath):
            self.download(self.URL, filepath)

