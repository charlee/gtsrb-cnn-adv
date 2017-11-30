import os
import numpy as np
from cnn_gtsrb.dataset.canvas import Canvas

def dump_sample(dirname, image_size, mode, filename):

    data = np.load(os.path.join('tmp', dirname, 'training.npy'))
    np.random.shuffle(data)

    c = Canvas(image_size, 10, 1, mode=mode)

    for i in range(10):
        image = data[data[:, -1] == i, :-1][0].astype(np.uint8)

        c.paste_np(image, i, 0)

    c.save(filename)

dump_sample('color_gtsrb_10_data\\gtsrb_data', 32, 'RGB', 'docs\\data-sample-cgtsrb10.png')
dump_sample('cifar10_data\\cifar10_data', 32, 'RGB', 'docs\\data-sample-cifar10.png')
dump_sample('fashion_mnist_data\\mnist_data', 28, 'L', 'docs\\data-sample-fmnist.png')
dump_sample('mnist_bg_data\\mnist_bg_data', 28, 'L', 'docs\\data-sample-mnistbg.png')