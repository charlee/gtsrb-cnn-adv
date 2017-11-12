import logging
import os
import numpy as np
from cnn_gtsrb.dataset.gtsrb import GtsrbProvider
from cnn_gtsrb.cnn.model import CNNModel
from cnn_gtsrb.dataset.canvas import Canvas

import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cnn_gtsrb.attacks.crafting import generate_adv_examples


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)


SAVE_DIR = os.path.join('tmp', 'gtsrb_adv_fgsm')
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)


gtsrb = GtsrbProvider(ignore_small=True)
# gtsrb.dump_images()

# Get data for each class in MNIST
data = gtsrb.raw_train_data()
np.random.shuffle(data)

grouped_data = []
for i in range(0, gtsrb.CLASSES, 2):
    grouped_data.append(data[(data[:,-1] == i), :])

images = np.concatenate(
    [np.expand_dims(group[0], 0) for group in grouped_data], axis=0
)


# Make CNN Model
cnn = CNNModel(
    image_size=gtsrb.IMAGE_SIZE,
    classes=gtsrb.CLASSES,
    model_name='gtsrb-32x32',
    model_dir='tmp/gtsrb_model-32x32',
    conv_layers=[32, 64, 128],
    fc_layer=512,
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
        num_classes=gtsrb.CLASSES,
        output_file=os.path.join(SAVE_DIR, 'adv_examples-{:02d}.png'.format(int(eps * 100)))
    )

    success_matrix.append(eps * m + 1 * ~m)

success_matrix = np.minimum.reduce(np.array(success_matrix))
success_matrix.dump(os.path.join(SAVE_DIR, 'success_rate.npy'))
print(success_matrix)


# cnn.test(gtsrb)
cnn.end_session()

#cnn.test(2000, gtsrb)

# for i in range(100):
#     data, label = gtsrb.next_batch('test')
#     print(data, label)


