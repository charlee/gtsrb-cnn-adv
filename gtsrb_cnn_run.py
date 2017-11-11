import logging
import tensorflow as tf
from cnn_gtsrb.dataset.gtsrb import GtsrbProvider
from cnn_gtsrb.cnn.model import CNNModel
#logging.basicConfig(level=logging.INFO)

tf.logging.set_verbosity(tf.logging.INFO)

gtsrb = GtsrbProvider()
# gtsrb.dump_images()

cnn = CNNModel(
    image_size=gtsrb.IMAGE_SIZE,
    classes=gtsrb.CLASSES,
    model_name='gtsrb-32x32',
    model_dir='tmp/gtsrb_model-32x32',
    conv_layers=[32, 64],
    fc_layer=1024,
)
cnn.train(20000, gtsrb)

#cnn.test(2000, gtsrb)

# for i in range(100):
#     data, label = gtsrb.next_batch('test')
#     print(data, label)

