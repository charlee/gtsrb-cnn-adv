import sys
import logging
import tensorflow as tf
from cnn_gtsrb.dataset.color_gtsrb_10 import ColorGtsrb10Provider
from cnn_gtsrb.cnn.model import CNNModel
#logging.basicConfig(level=logging.INFO)

tf.logging.set_verbosity(tf.logging.INFO)


def train_cnn_cgtsrb10():
    gtsrb = ColorGtsrb10Provider()
    # gtsrb.dump_images()

    cnn = CNNModel(
        image_size=gtsrb.IMAGE_SIZE,
        classes=gtsrb.CLASSES,
        channels=3,
        model_name='gtsrb-32x32',
        model_dir='tmp/cgtsrb10_model-32x32',
        conv_layers=[32, 64, 128],
        fc_layer=512,
    )

    x, y = cnn.make_inputs()
    probs = cnn.make_model(x)

    cnn.start_session()
    cnn.train(probs, x, y, 20000, gtsrb)
    # cnn.test(gtsrb)
    cnn.end_session()




if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'train_cnn_cgtsrb10':
        train_cnn_cgtsrb10()