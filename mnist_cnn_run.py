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
    model_name='mnist-28x28',
    model_dir='tmp/mnist_model-28x28',
    conv_layers=[32, 64],
    fc_layer=1028,
)
cnn.train(50000, mnist)

# cnn.test(10000, mnist)

# for i in range(100):
#     data, label = gtsrb.next_batch('test')
#     print(data, label)


