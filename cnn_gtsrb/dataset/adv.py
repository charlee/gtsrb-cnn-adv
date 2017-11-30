import glob
import numpy as np
from .base import DatasetProvider

class AdversarialProvider(DatasetProvider):

    """Load adversarial examples from given dir and use as dataset."""

    def __init__(self, adv_pattern):
        results = []
        for f in glob.glob(adv_pattern):
            results.append(np.load(f))

        data = np.concatenate(results, axis=0)

        self.IMAGE_SIZE = int(data[0][0])
        self.CLASSES = int(data[0][1])
        self.CHANNELS = int(data[0][2])

        size = int(self.IMAGE_SIZE * self.IMAGE_SIZE * self.CHANNELS)

        labels = data[:,6:7]
        images = data[:, 10+size:10+size+size]

        data = np.concatenate([images, labels], axis=1).astype(np.uint8)
        np.random.shuffle(data)
        train_size = int(data.shape[0] * 0.8);

        self._train_data = data[:train_size]
        self._test_data = data[train_size:]
        self.batch_size = 100

    def raw_train_data(self):
        return self._train_data

    def raw_test_data(self):
        return self._test_data


