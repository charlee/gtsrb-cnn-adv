import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Image size of each sample. Actual size is (w, h) = (IMAGE_SIZE, IMAGE_SIZE)
# Since we use 2 pooling layers each of which reduces image size to half,
# This value must be dividable by 4
IMAGE_SIZE = 32

# Number of classes.
CLASSES = 43

# Data temporary directory
DATA_TEMP_GTSRB = os.path.join(BASE_DIR, 'tmp', 'gtsrb_data')
DATA_TEMP_COLOR_GTSRB = os.path.join(BASE_DIR, 'tmp', 'color_gtsrb_data')
DATA_TEMP_COLOR_GTSRB_10 = os.path.join(BASE_DIR, 'tmp', 'color_gtsrb_10_data')
DATA_TEMP_MNIST = os.path.join(BASE_DIR, 'tmp', 'mnist_data')
DATA_TEMP_MNIST_BG = os.path.join(BASE_DIR, 'tmp', 'mnist_bg_data')
DATA_TEMP_FASHION_MNIST = os.path.join(BASE_DIR, 'tmp', 'fashion_mnist_data')
DATA_TEMP_CIFAR10 = os.path.join(BASE_DIR, 'tmp', 'cifar10_data')
DATA_TEMP_GEO10 = os.path.join(BASE_DIR, 'tmp', 'geo10_data')