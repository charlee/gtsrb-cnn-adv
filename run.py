import sys
import logging
# import tensorflow as tf
from cnn_gtsrb.dataset.color_gtsrb_10 import ColorGtsrb10Provider
from cnn_gtsrb.dataset.fashion_mnist import FashionMnistProvider
from cnn_gtsrb.dataset.cifar10 import Cifar10Provider
from cnn_gtsrb.cnn.model import CNNModel
#logging.basicConfig(level=logging.INFO)

# tf.logging.set_verbosity(tf.logging.INFO)


def train_cnn_cgtsrb10():
    gtsrb = ColorGtsrb10Provider()
    gtsrb.dump_images()

    cnn = CNNModel(
        image_size=gtsrb.IMAGE_SIZE,
        classes=gtsrb.CLASSES,
        channels=3,
        model_name='gtsrb-32x32',
        model_dir='tmp/cgtsrb10_model-32x32',
        conv_layers=[32, 64, 128],
        fc_layers=[512],
    )

    x, y = cnn.make_inputs()
    probs = cnn.make_model(x)

    cnn.start_session()
    cnn.train(probs, x, y, 20000, gtsrb)
    # cnn.test(gtsrb)
    cnn.end_session()


def train_cnn_cifar10():
    cifar10 = Cifar10Provider()
    #cifar10.dump_images()

    cnn = CNNModel(
        image_size=cifar10.IMAGE_SIZE,
        classes=cifar10.CLASSES,
        channels=3,
        kernel_size=[3, 3],
        model_name='cifar10-32x32',
        model_dir='tmp/cifar10_model-32x32',
        conv_layers=[48, 96, 192],
        fc_layers=[512, 256],
    )

    x, y = cnn.make_inputs()
    probs = cnn.make_model(x)

    cnn.start_session()
    cnn.train(probs, x, y, 20000, cifar10)
    # cnn.test(gtsrb)
    cnn.end_session()


def train_cnn_fashion_mnist():
    fmnist = FashionMnistProvider()
    # fmnist.dump_images()

    cnn = CNNModel(
        image_size=fmnist.IMAGE_SIZE,
        classes=fmnist.CLASSES,
        channels=fmnist.CHANNELS,
        model_name='fmnist-28x28',
        model_dir='tmp/fmnist_model-28x28',
        conv_layers=[32, 64],
        fc_layers=[1024],
    )

    x, y = cnn.make_inputs()
    probs = cnn.make_model(x)

    cnn.start_session()
    cnn.train(probs, x, y, 20000, fmnist)
    # cnn.test(gtsrb)
    cnn.end_session()



if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'train_cnn_cgtsrb10':
        train_cnn_cgtsrb10()
    elif cmd == 'train_cnn_cifar10':
        train_cnn_cifar10()
    elif cmd == 'train_cnn_fashion_mnist':
        train_cnn_fashion_mnist()