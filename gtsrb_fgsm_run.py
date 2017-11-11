import logging
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cnn_gtsrb.dataset.gtsrb import GtsrbProvider
from cnn_gtsrb.cnn.model import CNNModel
#logging.basicConfig(level=logging.INFO)

tf.logging.set_verbosity(tf.logging.INFO)

fgsm_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.}

gtsrb = GtsrbProvider()
# gtsrb.dump_images()

cnn = CNNModel(
    image_size=gtsrb.IMAGE_SIZE,
    classes=gtsrb.CLASSES,
    model_name='gtsrb-32x32.1',
    model_dir='tmp/gtsrb_model-32x32.1',
    conv_layers=[32, 64],
    fc_layer=1024,
)

x, y = cnn.make_inputs()

cnn.make_model(x, y)
cnn.start_session()

fgsm = FastGradientMethod(cnn, sess=cnn.sess)
adv_x = fgsm.generate(x, **fgsm_params)

cnn.calculate_accuracy(gtsrb)
# cnn.test(gtsrb)
cnn.end_session()

#cnn.test(2000, gtsrb)

# for i in range(100):
#     data, label = gtsrb.next_batch('test')
#     print(data, label)


