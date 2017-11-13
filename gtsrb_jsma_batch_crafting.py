import logging
import os
import numpy as np
import tensorflow as tf
from cnn_gtsrb.dataset.gtsrb import GtsrbProvider
from cnn_gtsrb.cnn.model import CNNModel

from cnn_gtsrb.attacks.crafting import BatchJSMACrafting


DIR_NAME = os.path.join('tmp', 'batch_adv')
if not os.path.isdir(DIR_NAME):
    os.makedirs(DIR_NAME)


logging.basicConfig(level=logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)


gtsrb = GtsrbProvider(ignore_small=True)

# Make CNN Model
cnn = CNNModel(
    image_size=gtsrb.IMAGE_SIZE,
    classes=gtsrb.CLASSES,
    model_name='gtsrb-32x32',
    model_dir='tmp/gtsrb_model-32x32',
    conv_layers=[32, 64, 128],
    fc_layer=512,
)

jsma_params = {'theta': 1., 'gamma': 0.1, 'clip_min': 0., 'clip_max': 1., 'y_target': None}

test_data = gtsrb.raw_test_data()
test_data = test_data[:5000]

jsma_crafting = BatchJSMACrafting(cnn, jsma_params, gtsrb.IMAGE_SIZE, gtsrb.CLASSES)


cnn.start_session()
cnn.init_session_and_restore()

BATCH_SIZE=100
for theta in np.arange(0.2, 0.7, 0.2):
    jsma_crafting.update_params({'theta': theta})

    for batch_pos in range(0, test_data.shape[0], BATCH_SIZE):

        print("================ theta = {}, batch = {} =============".format(theta, batch_pos))
        batch = test_data[batch_pos:batch_pos+BATCH_SIZE]
        result = jsma_crafting.craft_examples(batch)
        result = jsma_crafting.summarize(batch, *result)

        result.dump(os.path.join(DIR_NAME, 'jsma_gtsrb-{:0.1f}-{}.npy'.format(theta, batch_pos)))

cnn.end_session()