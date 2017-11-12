import logging
import os
import numpy as np
from cnn_gtsrb.dataset.mnist import MnistProvider
from cnn_gtsrb.cnn.model import CNNModel
from cnn_gtsrb.dataset.canvas import Canvas

import tensorflow as tf
from cleverhans.attacks import SaliencyMapMethod
from cnn_gtsrb.attacks.crafting import generate_adv_examples


SAVE_DIR = os.path.join('tmp', 'mnist_adv_jsma')
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)


mnist = MnistProvider()
# mnist.dump_images()

# Get data for each class in MNIST
data = mnist.raw_train_data()

grouped_data = []
for i in range(0, mnist.CLASSES):
    grouped_data.append(data[(data[:,-1] == i), :])

images = np.concatenate(
    [np.expand_dims(group[0], 0) for group in grouped_data], axis=0
)


# Make CNN Model
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
cnn.init_session_and_restore()

jsma = SaliencyMapMethod(cnn, sess=cnn.sess)
jsma_params = {'theta': 1., 'gamma': 0.1, 'clip_min': 0., 'clip_max': 1., 'y_target': None}


generate_adv_examples(
    attack=jsma, 
    attack_params=jsma_params,
    cnn=cnn,
    probs=probs,
    x=x,
    images=images,
    num_classes=mnist.CLASSES,
    output_file=os.path.join(SAVE_DIR, 'adv_examples.png')
)


# cnn.test(mnist)
cnn.end_session()

#cnn.test(2000, mnist)

# for i in range(100):
#     data, label = gtsrb.next_batch('test')
#     print(data, label)


