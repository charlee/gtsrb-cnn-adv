import os
import sys
import math
import zipfile
import numpy as np
import urllib.request
from PIL import Image, ImageDraw


class DatasetProvider():

    IMAGE_SIZE = 28
    CHANNELS = 1
    CLASSES = 10

    def __init__(self, batch_size=100, *args, **kwargs):
        self.batch_size = batch_size
        self.init(*args, **kwargs)

    def init(self):
        raise NotImplemented('Inherited class must implement `init`')

    def next_batch(self, type='training'):
        raise NotImplemented('Inherited class must implement `next_batch`')

    def download(self, url, local_file):

        target_dir = os.path.dirname(local_file)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        status = {'total_m': 0}
        def reporthook(count, block_size, total_size):
            m = count * block_size // 1048756
            print('.' * (m - status['total_m']), end='')
            sys.stdout.flush()
            status['total_m'] = m

        print('Downloading from {} to {}'.format(url, local_file))

        tmp_file = '{}.download'.format(local_file)
        urllib.request.urlretrieve(url, tmp_file, reporthook=reporthook)
        print('done.')
        sys.stdout.flush()

        os.rename(tmp_file, local_file)

    def unzip(self, local_file, dest_dir):
        print('Unzipping {} to {}...'.format(local_file, dest_dir), end='')
        sys.stdout.flush()
        zip_ref = zipfile.ZipFile(local_file, 'r')
        zip_ref.extractall(dest_dir)
        zip_ref.close()
        print('done.')
        sys.stdout.flush()

    def download_and_unzip(self, url, local_file, data_dir):
        """Will download url to local_file (if local_file does not exist)
        and extract local_file to data_dir.
        However will do nothing if data_dir exists.
        """
        if not os.path.isdir(data_dir):
            if not os.path.isfile(local_file):
                self.download(url, local_file)
            os.makedirs(data_dir)
            self.unzip(local_file, data_dir)

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

                if self.CHANNELS == 1:
                    mode = 'L'
                else:
                    mode = 'RGB'
                im = Image.new(mode, (canvas_width, canvas_height))
                d = ImageDraw.Draw(im)
                d.text((10, 10), 'Class = {}'.format(class_id), fill=255)
                del d

                for idx in range(images_in_class.shape[0]):
                    image = images_in_class[idx]
                    image = np.reshape(image, [self.IMAGE_SIZE, self.IMAGE_SIZE, self.CHANNELS]).astype(np.uint8)
                    im2 = Image.fromarray(image)

                    x = idx % 10 * self.IMAGE_SIZE
                    y = idx // 10 * self.IMAGE_SIZE + 30

                    im.paste(im2, (x, y))
                    im2.close()

                filename = '{}-dump.class{}.png'.format(t, class_id)
                print('Dumpping ({}, {}) to {}'.format(t, class_id, filename))
                im.save(os.path.join(self.data_dir, filename))
                im.close()

    def raw_train_data(self):
        """Return unprocessed data."""
        return np.load(os.path.join(self.data_dir, 'training.npy'))

    def raw_test_data(self):
        return np.load(os.path.join(self.data_dir, 'test.npy'))

    def train_data(self):
        """Return all train images and labels."""
        data = np.load(os.path.join(self.data_dir, 'training.npy'))
        np.random.shuffle(data)
        return self.split_images_and_labels(size and data[:size] or data)

    def next_batch(self, batch_size=None):

        if not batch_size:
            batch_size = self.batch_size

        if not hasattr(self, 'pool'):
            self.pool = np.load(os.path.join(self.data_dir, 'training.npy'))
            self.current_pos = 0
            np.random.shuffle(self.pool)

        batch = self.pool[self.current_pos:(self.current_pos+batch_size)]
        if len(batch) < batch_size:
            batch_ = self.pool[0:(batch_size - len(batch))]
            batch = np.append(batch, batch_, axis=0)
            self.current_pos = batch_size - len(batch)
        else:
            self.current_pos += batch_size

        return self.split_images_and_labels(batch)

    def test_data(self, size=None):
        """Pick `size` test data randomly."""
        data = np.load(os.path.join(self.data_dir, 'test.npy'))
        np.random.shuffle(data)

        if size is not None:
            data = data[:size]

        return self.split_images_and_labels(data)

    def split_images_and_labels(self, data):
        # Cut data to image data and label data
        images = data[:, :-1]
        labels = data[:, -1:]

        # Normalize image data to [0.0, 1.0] and reshape to [batch, w, h, channels]
        images = np.reshape(images * (1. / 255), [-1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.CHANNELS])

        # Generate one hot vector for labels
        one_hot = np.zeros(shape=[labels.shape[0], self.CLASSES], dtype=np.float32)
        for i in range(labels.shape[0]):
            class_id = labels[i][0]
            one_hot[i, class_id] = 1

        return (images, one_hot)

