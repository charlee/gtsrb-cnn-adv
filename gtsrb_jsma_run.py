import logging
import tensorflow as tf
from cleverhans.attacks import SaliencyMapMethod
from cnn_gtsrb.dataset.gtsrb import GtsrbProvider
from cnn_gtsrb.cnn.model import CNNModel
#logging.basicConfig(level=logging.INFO)

tf.logging.set_verbosity(tf.logging.INFO)

jsma_params = {'theta': 1., 'gamma': 0.1, 'clip_min': 0., 'clip_max': 1., 'y_target': None}

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

fgsm = SaliencyMapMethod(cnn, sess=cnn.sess)
adv_x = fgsm.generate(x, **jsma_params)
probs = cnn.make_model(adv_x)

cnn.adv_test(probs, x, y, adv_x, gtsrb.test_data(size=10))
# cnn.test(gtsrb)
cnn.end_session()

#cnn.test(2000, gtsrb)

# for i in range(100):
#     data, label = gtsrb.next_batch('test')
#     print(data, label)


