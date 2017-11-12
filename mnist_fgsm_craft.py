import logging
import os
import numpy as np
from cnn_gtsrb.dataset.mnist import MnistProvider
from cnn_gtsrb.cnn.model import CNNModel
from cnn_gtsrb.dataset.canvas import Canvas

import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cnn_gtsrb.attacks.crafting import generate_adv_examples


SAVE_DIR = os.path.join('tmp', 'mnist_adv_fgsm')
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)


mnist = MnistProvider()
# mnist.dump_images()

# Get data for each class in MNIST
data = mnist.raw_train_data()

grouped_data = []
for i in range(mnist.CLASSES):
    grouped_data.append(data[(data[:,-1] == i), :-1])

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

fgsm = FastGradientMethod(cnn, sess=cnn.sess)

success_matrix = []

for eps in np.concatenate((np.arange(0.02, 0.1, 0.02), np.arange(0.1, 1, 0.1))):
    m = generate_adv_examples(
        attack=fgsm, 
        attack_params={'eps': eps, 'clip_min': 0., 'clip_max': 1.},
        cnn=cnn,
        probs=probs,
        x=x,
        images=images,
        actual_classes=mnist.CLASSES,
        output_file=os.path.join(SAVE_DIR, 'adv_examples-{:02d}.png'.format(int(eps * 100)))
    )

    success_matrix.append(eps * m + 1 * ~m)

success_matrix = np.minimum.reduce(np.array(success_matrix))
success_matrix.dump(os.path.join(SAVE_DIR, 'success_rate.npy'))
print(success_matrix)


# cnn.test(mnist)
cnn.end_session()

#cnn.test(2000, mnist)

# for i in range(100):
#     data, label = gtsrb.next_batch('test')
#     print(data, label)


