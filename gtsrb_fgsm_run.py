import logging
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cnn_gtsrb.dataset.gtsrb import GtsrbProvider
from cnn_gtsrb.cnn.model import CNNModel
#logging.basicConfig(level=logging.INFO)

tf.logging.set_verbosity(tf.logging.INFO)

fgsm_params = {'eps': 0.5, 'clip_min': 0., 'clip_max': 1.}

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

fgsm = FastGradientMethod(cnn, sess=cnn.sess)
adv_x = fgsm.generate(x, **fgsm_params)
probs = cnn.make_model(adv_x)

cnn.adv_test(probs, x, y, adv_x, gtsrb.test_data(size=1000))
# cnn.test(gtsrb)
cnn.end_session()

#cnn.test(2000, gtsrb)

# for i in range(100):
#     data, label = gtsrb.next_batch('test')
#     print(data, label)


