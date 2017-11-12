import logging
import os
import numpy as np
from cnn_gtsrb.dataset.mnist import MnistProvider
from cnn_gtsrb.cnn.model import CNNModel
from cnn_gtsrb.dataset.canvas import Canvas

import tensorflow as tf
from cleverhans.attacks import FastGradientMethod


SAVE_DIR = os.path.join('tmp', 'mnist_adv_fgsm')
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def generate_adv_examples(fgsm, probs, eps, image_size, classes, grouped_data):

    canvas = Canvas(image_size, classes, classes)
    success_matrix = np.ones([classes, classes])
    fgsm_params = {'eps': eps, 'clip_min': 0., 'clip_max': 1.}

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



mnist = MnistProvider()
# mnist.dump_images()


# Get data for each class in MNIST
data = mnist.raw_train_data()

grouped_data = []
for i in range(mnist.CLASSES):
    grouped_data.append(data[(data[:,-1] == i), :-1])


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


success_matrices = [
    generate_adv_examples(fgsm, probs, eps, mnist.IMAGE_SIZE, mnist.CLASSES, grouped_data)
    for eps in np.concatenate((np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1)))
]

success_rates = np.min([np.expand_dims(m, 0) for m in success_matrices], axis=0)
print(success_rates[0])


# cnn.test(mnist)
cnn.end_session()

#cnn.test(2000, mnist)

# for i in range(100):
#     data, label = gtsrb.next_batch('test')
#     print(data, label)


