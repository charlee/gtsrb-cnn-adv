import logging
import os
import numpy as np
from cnn_gtsrb.dataset.gtsrb import GtsrbProvider
from cnn_gtsrb.cnn.model import CNNModel
from cnn_gtsrb.dataset.canvas import Canvas

import tensorflow as tf
from cleverhans.attacks import SaliencyMapMethod


SAVE_DIR = os.path.join('tmp', 'gtsrb_adv_jsma')
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def generate_adv_examples(fgsm, probs, image_size, classes, grouped_data):

    canvas = Canvas(image_size, classes, classes)
    success_matrix = np.ones([classes, classes])
    jsma_params = {'theta': 1., 'gamma': 0.1, 'clip_min': 0., 'clip_max': 1., 'y_target': None}

    print('Generating adversarial examples with FGSM, eps={}'.format(eps))

    # Generate adversarial examples for each class
    for class_id in range(classes):
        # Get an image
        x_val = grouped_data[class_id][1]

        # normalize
        x_val = np.reshape(x_val * (1. / 255), [1, image_size, image_size, 1])

        # Make example for each target class
        for target in range(classes):
            # Skip same class
            if target == class_id:
                continue

            one_hot_target = np.zeros([1, classes], dtype=np.float32)
            one_hot_target[0, target] = 1.
            fgsm_params['y_target'] = one_hot_target
            adv_x_val = fgsm.generate_np(x_val, **fgsm_params)

            im_data = (adv_x_val * 255).astype(np.uint8)
            canvas.paste_np(im_data, target, class_id)

            # Evaluate
            predict = cnn.sess.run(tf.argmax(probs, axis=1), feed_dict={x:adv_x_val})

            if predict[0] == target:
                success_matrix[class_id, target] = eps
        
    filename = 'adv_examples-{:02d}.png'.format(int(eps * 100))
    canvas.save(os.path.join(SAVE_DIR, filename))
    canvas.close()

    return success_matrix



gtsrb = GtsrbProvider(ignore_small=True)
# gtsrb.dump_images()


# Get data for each class in MNIST
data = gtsrb.raw_train_data()

grouped_data = []
for i in range(0, gtsrb.CLASSES, 5):
    grouped_data.append(data[(data[:,-1] == i), :-1])


# Make CNN Model
cnn = CNNModel(
    image_size=gtsrb.IMAGE_SIZE,
    classes=gtsrb.CLASSES,
    model_name='gtsrb-28x28.1',
    model_dir='tmp/gtsrb_model-28x28.1',
    conv_layers=[32, 64],
    fc_layer=1024,
)

x, y = cnn.make_inputs()

probs = cnn.make_model(x)
cnn.start_session()
cnn.init_session_and_restore()

fgsm = FastGradientMethod(cnn, sess=cnn.sess)


success_matrices = [
    generate_adv_examples(fgsm, probs, eps, gtsrb.IMAGE_SIZE, gtsrb.CLASSES, grouped_data)
    for eps in np.concatenate((np.arange(0.02, 0.1, 0.02), np.arange(0.1, 1, 0.1)))
]

success_rates = np.min([np.expand_dims(m, 0) for m in success_matrices], axis=0)
success_rates[0].dump(os.path.join(SAVE_DIR, 'success_rate.npy'))


# cnn.test(gtsrb)
cnn.end_session()

#cnn.test(2000, gtsrb)

# for i in range(100):
#     data, label = gtsrb.next_batch('test')
#     print(data, label)


