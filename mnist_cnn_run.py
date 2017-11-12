import logging
import tensorflow as tf
from cnn_gtsrb.dataset.mnist import MnistProvider
from cnn_gtsrb.cnn.model import CNNModel
#logging.basicConfig(level=logging.INFO)

tf.logging.set_verbosity(tf.logging.INFO)

mnist = MnistProvider()
# mnist.dump_images()

cnn = CNNModel(
    image_size=mnist.IMAGE_SIZE,
    classes=mnist.CLASSES,
    model_name='mnist-28x28.1',
    model_dir='tmp/mnist_model-28x28.1',
    conv_layers=[32, 64],
    fc_layer=1024,
)

x, y = cnn.make_inputs()

probs = cnn.make_model(x)

cnn.start_session()
cnn.train(probs, x, y, 3000, mnist)
# cnn.test(gtsrb)
cnn.end_session()

