from cnn_gtsrb import settings
from .mnist import MnistProvider


class FashionMnistProvider(MnistProvider):
    name = 'fmnist'
    URL_BASE = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    DATA_DIR = settings.DATA_TEMP_FASHION_MNIST