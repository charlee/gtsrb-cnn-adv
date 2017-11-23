"""
Colored GTSRB dataset.

Dataset output will be an nparray dump with shape of [?, 32 * 32 * 3].
"""

import os
import numpy as np
from PIL import Image, ImageOps
from .gtsrb import GtsrbProvider, GtsrbClass
from cnn_gtsrb import settings

class ColorGtsrbClass(GtsrbClass):
    def read_image(self, image_info):
        """Read from ppm image and return a ndarray with size [1, self.IMAGE_SIZE * self.IMAGE_SIZE + 1],
        in which the last element is the label."""
        im = Image.open(image_info['path'])
        im = im.crop(image_info['box'])

        if self.ignore_small and (im.size[0] < self.IMAGE_SIZE or im.size[1] < self.IMAGE_SIZE):
            return None

        im = im.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
        im = ImageOps.equalize(im)
        data = im.tobytes()
        data = np.fromstring(data, dtype=np.uint8)
        data = np.append(data, [image_info['label']])

        size = data.shape[0]

        data = np.reshape(data, [1, size])

        return data


class ColorGtsrbProvider(GtsrbProvider):

    CHANNELS = 3
    DATA_DIR = settings.DATA_TEMP_COLOR_GTSRB

    def process_gtsrb(self, path, prefix):
        """Pre-process gtsrb data."""

        all_dataset = []

        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.csv'):
                    print('Generating dataset for {}'.format(f))
                    filepath = os.path.join(root, f)
                    gtsrb_class = ColorGtsrbClass(filepath, ignore_small=self.ignore_small)
                    dataset = gtsrb_class.get_dataset()

                    all_dataset += dataset

        all = np.concatenate(all_dataset, axis=0)
        all.dump(os.path.join(self.data_dir, '{}.npy'.format(prefix)))