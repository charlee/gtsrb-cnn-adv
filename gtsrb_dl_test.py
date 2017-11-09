import logging
import tensorflow as tf
from cnn_gtsrb.dataset.gtsrb import GtsrbProvider
from cnn_gtsrb.cnn.model import CNNModel
#logging.basicConfig(level=logging.INFO)

tf.logging.set_verbosity(tf.logging.INFO)

gtsrb = GtsrbProvider()

cnn = CNNModel(image_size=gtsrb.IMAGE_SIZE, classes=gtsrb.CLASSES)
cnn.train(10000, gtsrb)

# for i in range(100):
#     data, label = gtsrb.next_batch()
#
# print(label[0])
