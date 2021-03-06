import os
import shutil
import logging
import csv
import numpy as np
import cnn_gtsrb.settings as settings
from PIL import Image, ImageOps
from .base import DatasetProvider


logger = logging.getLogger('gtsrb')


class GtsrbClass:
    """A GTSRB class folder with images of the same class."""

    IMAGE_SIZE = 32

    def __init__(self, index_file, ignore_small=False):

        self.image_list = []
        self.ignore_small = ignore_small

        dir = os.path.dirname(index_file)
        basename = os.path.basename(index_file)
        self.index_basename = os.path.splitext(basename)[0]

        with open(index_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=';')
            for (filename, width, height, x1, y1, x2, y2, class_id) in csv_reader:
                if filename == 'Filename':
                    continue

                image_path = os.path.join(dir, filename)

                if os.path.isfile(image_path):
                    self.image_list.append({
                        'path': image_path,
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'label': int(class_id),
                    })

    def read_image(self, image_info):
        """Read from ppm image and return a ndarray with size [1, self.IMAGE_SIZE * self.IMAGE_SIZE + 1],
        in which the last element is the label."""
        im = Image.open(image_info['path'])
        im = im.crop(image_info['box'])

        if self.ignore_small and (im.size[0] < self.IMAGE_SIZE or im.size[1] < self.IMAGE_SIZE):
            return None

        im = im.convert('L')
        im = im.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
        im = ImageOps.equalize(im)
        data = im.tobytes()
        data = np.fromstring(data, dtype=np.uint8)
        data = np.append(data, [image_info['label']])

        size = data.shape[0]

        data = np.reshape(data, [1, size])

        return data

    def get_dataset(self):
        """Use first 80% as traning set and last 80% as test set.
        GTSRB has 30 images as a group for each traffic sign.
        In each group, images are from the same real life sign (screenshot from a video).
        In order to make sure test data won't appear in traning data,
        we take 80% of total distinct traffic signs as traning set.
        """
        images = [self.read_image(im) for im in self.image_list]
        images = [im for im in images if im is not None]

        return images


class GtsrbProvider(DatasetProvider):
    name = 'gtsrb'
    TRAINING_URL = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
    TEST_URL = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip'
    TEST_ANNOTATION_URL = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip'
    DATA_DIR = settings.DATA_TEMP_GTSRB

    IMAGE_SIZE = 32
    CLASSES = 43

    def init(self, ignore_small=False):

        if ignore_small:
            dirname = 'gtsrb_data.nosmall'
        else:
            dirname = 'gtsrb_data'

        self.ignore_small = ignore_small

        self.data_dir = os.path.join(self.DATA_DIR, dirname)
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)

            rawdata_dir = os.path.join(self.DATA_DIR, 'rawdata')

            training_dir = os.path.join(rawdata_dir, 'training')
            self.download_and_unzip(
                self.TRAINING_URL,
                os.path.join(rawdata_dir, 'GTSRB_Final_Training_Images.zip'),
                training_dir,
            )

            test_dir = os.path.join(rawdata_dir, 'test')
            self.download_and_unzip(
                self.TEST_URL,
                os.path.join(rawdata_dir, 'GTSRB_Final_Test_Images.zip'),
                test_dir,
            )

            test_annotation_dir = os.path.join(rawdata_dir, 'test-annotation')
            self.download_and_unzip(
                self.TEST_ANNOTATION_URL,
                os.path.join(rawdata_dir, 'GTSRB_Final_Test_GT.zip'),
                test_annotation_dir,
            )

            useless_annotation = os.path.join(test_dir, 'GTSRB', 'Final_Test', 'Images', 'GT-final_test.test.csv')
            if os.path.isfile(useless_annotation):
                os.remove(useless_annotation)

            useful_annotation = os.path.join(test_dir, 'GTSRB', 'Final_Test', 'Images', 'GT-final_test.csv')

            if not os.path.isfile(useful_annotation):
                shutil.copyfile(
                    os.path.join(test_annotation_dir, 'GT-final_test.csv'),
                    useful_annotation,
                )

            self.process_gtsrb(training_dir, 'training')
            self.process_gtsrb(test_dir, 'test')

    def process_gtsrb(self, path, prefix):
        """Pre-process gtsrb data."""

        all_dataset = []

        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.csv'):
                    print('Generating dataset for {}'.format(f))
                    filepath = os.path.join(root, f)
                    gtsrb_class = GtsrbClass(filepath, ignore_small=self.ignore_small)
                    dataset = gtsrb_class.get_dataset()

                    all_dataset += dataset

        all = np.concatenate(all_dataset, axis=0)
        all.dump(os.path.join(self.data_dir, '{}.npy'.format(prefix)))
