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
    model_name='gtsrb-64x64',
    model_dir='tmp/gtsrb_model-64x64',
    conv_layers=[32, 64, 128],
    fc_layer=512,
)

x, y = cnn.make_inputs()
probs = cnn.make_model(x)

cnn.start_session()
cnn.train(probs, x, y, 10000, gtsrb)
# cnn.test(gtsrb)
cnn.end_session()

#cnn.test(2000, gtsrb)

# for i in range(100):
#     data, label = gtsrb.next_batch('test')
#     print(data, label)

