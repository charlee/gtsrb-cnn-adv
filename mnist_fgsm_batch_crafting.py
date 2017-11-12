import logging
import os
import numpy as np
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cnn_gtsrb.dataset.mnist import MnistProvider
from cnn_gtsrb.cnn.model import CNNModel

from cnn_gtsrb.attacks.crafting import BatchFGSMCrafting

DIR_NAME = os.path.join('tmp', 'batch_adv')
if not os.path.isdir(DIR_NAME):
    os.makedirs(DIR_NAME)

logging.basicConfig(level=logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)

fgsm_params = {'eps': 0.2, 'clip_min': 0., 'clip_max': 1.}

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

test_data = mnist.raw_test_data()
test_data = test_data[:10]

fgsm_crafting = BatchFGSMCrafting(cnn, fgsm_params, mnist.IMAGE_SIZE, mnist.CLASSES)


cnn.start_session()
cnn.init_session_and_restore()

for i, eps in enumerate(np.arange(0.05, 0.5, 0.05)):
    fgsm_crafting.update_params({'eps': eps})

    result = fgsm_crafting.craft_examples(test_data)
    result = fgsm_crafting.summarize(test_data, *result)

    result.dump(os.path.join(DIR_NAME, 'fgsm_mnist-{}.npy'.format(i)))

cnn.end_session()