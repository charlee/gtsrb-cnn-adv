import os
import logging
import csv
import numpy as np
import cnn_gtsrb.settings as settings
from PIL import Image
from .base import DatasetProvider


logger = logging.getLogger('gtsrb')


class GtsrbClass:
    """A GTSRB class folder with images of the same class."""

    IMAGE_SIZE = 32

    def __init__(self, index_file, train_ratio=80):
        self.train_ratio = train_ratio

        self.image_list = []
        self.training_set = []
        self.test_set = []

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
                    })

                    self.class_id = int(class_id)

    def read_image(self, image_info):
        im = Image.open(image_info['path'])
        im = im.crop(image_info['box'])
        im = im.convert('L')
        im = im.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
        data = im.tobytes()
        im.close()

        return data


    def datasets(self):
        """Use first 80% as traning set and last 80% as test set.
        GTSRB has 30 images as a group for each traffic sign.
        In each group, images are from the same real life sign (screenshot from a video).
        In order to make sure test data won't appear in traning data,
        we take 80% of total distinct traffic signs as traning set.
        """
        images = [self.read_image(im) for im in self.image_list]
        num_signs = len(images) // 30           # Number of distinct traffic signs

        training_count = min(int(num_signs * 0.8), num_signs - 1)

        training_set = images[:training_count * 30]
        test_set = images[training_count * 30:]

        # convert to ndarary
        array_size = self.IMAGE_SIZE * self.IMAGE_SIZE
        training_set = [np.fromstring(image, dtype=np.uint8, count=array_size)
                        for image in training_set]
        training_set = np.concatenate(training_set, axis=0)

        test_set = [np.fromstring(image, dtype=np.uint8, count=array_size)
                    for image in test_set]
        test_set = np.concatenate(test_set, axis=0)

        return (training_set, test_set, self.class_id)


class GtsrbProvider(DatasetProvider):
    DL_URL = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
    ZIP_NAME = 'GTSRB_Final_Training_Images.zip'
    DATA_DIR = settings.DATA_TEMP_GTSRB

    IMAGE_SIZE = 32
    CLASSES = 43

    def init(self):

        self.rawdata_dir = os.path.join(self.DATA_DIR, 'rawdata')
        self.data_dir = os.path.join(self.DATA_DIR, 'data')

        if not os.path.isdir(self.data_dir):
            if not os.path.isdir(self.rawdata_dir):

                local_file = os.path.join(self.DATA_DIR, self.ZIP_NAME)
                if not os.path.isfile(local_file):
                    self.download(self.DL_URL, local_file)

                self.unzip(local_file, self.rawdata_dir)

            os.makedirs(self.data_dir)
            self.process_gtsrb()

    def process_gtsrb(self):
        """Pre-process gtsrb data."""
        for root, dirs, files in os.walk(self.rawdata_dir):
            for f in files:
                if f.endswith('.csv'):
                    logger.info('Generating examples for {}'.format(f))
                    filepath = os.path.join(root, f)
                    gtsrb_class = GtsrbClass(filepath)
                    (training_set, test_set, class_label) = gtsrb_class.datasets()

                    training_set.dump(os.path.join(self.data_dir, 'training-{}.npy'.format(class_label)))
                    test_set.dump(os.path.join(self.data_dir, 'test-{}.npy'.format(class_label)))

    def next_batch(self, type='training'):

        array_size = self.IMAGE_SIZE * self.IMAGE_SIZE

        if not hasattr(self, 'pool'):
            self.pool = np.empty(shape=[0, array_size], dtype=np.uint8)
            self.current_file_no = 0

        # if local pool is less than batch_size then load new data
        while self.pool.shape[0] < self.batch_size:
            filename = '{}-{}.npy'.format(type, self.current_file_no)
            logger.info('load new file: {}'.format(filename))
            filepath = os.path.join(self.data_dir, filename)
            data = np.load(filepath)
            self.current_file_no += 1

            # If read all files then repeat
            if self.current_file_no >= self.CLASSES:
                self.current_file_no = 0

            data = np.reshape(data, [-1, array_size])
            self.pool = np.concatenate((self.pool, data), axis=0)

        np.random.shuffle(self.pool)

        batch = self.pool[0:self.batch_size]
        self.pool = self.pool[self.batch_size:]

        return batch

