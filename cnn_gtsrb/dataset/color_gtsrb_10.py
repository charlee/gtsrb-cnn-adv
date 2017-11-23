"""
Colored GTSRB dataset with only 10 classes.

Dataset output will be an nparray dump with shape of [?, 32 * 32 * 3].
"""

import os
import numpy as np
from cnn_gtsrb import settings
from .color_gtsrb import ColorGtsrbProvider, ColorGtsrbClass


class ColorGtsrb10Provider(ColorGtsrbProvider):

    CLASSES = 10
    DATA_DIR = settings.DATA_TEMP_COLOR_GTSRB_10

    def process_gtsrb(self, path, prefix):
        """Pre-process gtsrb data."""

        all_dataset = []

        chosen_classes = [2, 11, 17, 19, 28, 31, 35, 38, 41, 42]
        chosen_map = {v:i for i, v in enumerate(chosen_classes)} 

        chosen_indecis = ['GT-000{:02d}.csv'.format(i) for i in chosen_classes] + ['GT-final_test.csv']

        for root, dirs, files in os.walk(path):
            for f in files:
                if f in chosen_indecis:
                    print('Generating dataset for {}'.format(f))
                    filepath = os.path.join(root, f)
                    gtsrb_class = ColorGtsrbClass(filepath, ignore_small=self.ignore_small)
                    dataset = gtsrb_class.get_dataset()

                    # Filter only chosen classes
                    dataset = [d for d in dataset if d[0][-1] in chosen_classes]

                    # Replace class id
                    for d in dataset:
                        d[0][-1] = chosen_map[d[0][-1]]

                    all_dataset += dataset

        all = np.concatenate(all_dataset, axis=0)
        all.dump(os.path.join(self.data_dir, '{}.npy'.format(prefix)))
