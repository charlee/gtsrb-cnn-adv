import os
import math
import shutil
import logging
import csv
import numpy as np
import cnn_gtsrb.settings as settings
from PIL import Image, ImageOps, ImageDraw
from .base import DatasetProvider


logger = logging.getLogger('gtsrb')


class GtsrbClass:
    """A GTSRB class folder with images of the same class."""

    IMAGE_SIZE = 64

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
                        'label': int(class_id),
                    })

    def read_image(self, image_info):
        """Read from ppm image and return a ndarray with size [1, self.IMAGE_SIZE * self.IMAGE_SIZE + 1],
        in which the last element is the label."""
        im = Image.open(image_info['path'])
        im = im.crop(image_info['box'])
        im = im.convert('L')
        im = im.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
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

        return images


class GtsrbProvider(DatasetProvider):
    TRAINING_URL = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
    TEST_URL = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip'
    TEST_ANNOTATION_URL = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip'
    DATA_DIR = settings.DATA_TEMP_GTSRB

    IMAGE_SIZE = 64
    CLASSES = 43

    def init(self):

        self.data_dir = os.path.join(self.DATA_DIR, 'gtsrb_data')
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
                    gtsrb_class = GtsrbClass(filepath)
                    dataset = gtsrb_class.get_dataset()

                    all_dataset += dataset

        all = np.concatenate(all_dataset, axis=0)
        all.dump(os.path.join(self.data_dir, '{}.npy'.format(prefix)))

    def dump_images(self):
        """Dump npy data into png images."""
        for t in ('training', 'test'):
            filepath = os.path.join(self.data_dir, '{}.npy'.format(t))
            print('Dumping images from {}...'.format(filepath))
            images = np.load(filepath)

            # Sort images by label (the last field on axis 1)
            images = images[np.argsort(images[:,-1])]
            labels = images[:,-1]

            for class_id in range(self.CLASSES):
                images_in_class = images[(labels == class_id), :-1]
                canvas_width = self.IMAGE_SIZE * 10
                canvas_height = self.IMAGE_SIZE * math.ceil(images_in_class.shape[0] / 10) + 30

                im = Image.new('L', (canvas_width, canvas_height))
                d = ImageDraw.Draw(im)
                d.text((10, 10), 'Class = {}'.format(class_id), fill=255)
                del d

                for idx in range(images_in_class.shape[0]):
                    image = images_in_class[idx]
                    image = np.reshape(image, [self.IMAGE_SIZE, self.IMAGE_SIZE])
                    im2 = Image.fromarray(image)

                    x = idx % 10 * self.IMAGE_SIZE
                    y = idx // 10 * self.IMAGE_SIZE + 30

                    im.paste(im2, (x, y))
                    im2.close()

                filename = '{}-dump.class{}.png'.format(t, class_id)
                print('Dumpping ({}, {}) to {}'.format(t, class_id, filename))
                im.save(os.path.join(self.data_dir, filename))
                im.close()

    def next_batch(self, type='training', batch_size=None):

        image_size = self.IMAGE_SIZE * self.IMAGE_SIZE
        array_size = image_size + 1

        if not batch_size:
            batch_size = self.batch_size

        if not hasattr(self, 'pool'):
            self.pool = {
                'training': np.empty(shape=[0, array_size], dtype=np.uint8),
                'test': np.empty(shape=[0, array_size], dtype=np.uint8),
            }
            self.current_pos = 0

            self.pool['training'] = np.load(os.path.join(self.data_dir, 'training.npy'))
            self.pool['test'] = np.load(os.path.join(self.data_dir, 'test.npy'))

            np.random.shuffle(self.pool['training'])
            np.random.shuffle(self.pool['test'])

        batch = self.pool[type][self.current_pos:(self.current_pos+batch_size)]
        if len(batch) < batch_size:
            batch_ = self.pool[type][0:(batch_size - len(batch))]
            batch = np.append(batch, batch_, axis=0)
            self.current_pos = batch_size - len(batch)
        else:
            self.current_pos += batch_size

        # Cut data to image data and label data
        images = batch[:, :image_size]
        labels = batch[:, image_size:]

        # Convert label data to one-hot array
        one_hot_labels = np.zeros(shape=[labels.shape[0], self.CLASSES], dtype=np.float32)
        for i in range(labels.shape[0]):
            class_id = labels[i][0]
            one_hot_labels[i, class_id] = 1

        return (images, one_hot_labels)

