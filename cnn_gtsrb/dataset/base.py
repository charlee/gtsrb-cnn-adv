import os
import sys
import zipfile
import urllib.request


class DatasetProvider():

    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.init()

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
