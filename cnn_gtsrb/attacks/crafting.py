import logging
import numpy as np
import math
import tensorflow as tf
from cnn_gtsrb.dataset.canvas import Canvas

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)


def generate_adv_examples(
    attack, attack_params,
    cnn, probs, x,
    num_classes,
    images,
    output_file
    ):

    """
    Border colors:
    - red = identity
    - green = targeted attack success
    - blue = targeted attack failure but confusion success
    """

    logger.info('Generating adversarial examples, params={}'.format(attack_params))

    image_size = int(math.sqrt(images.shape[1]))
    class_list = images[:,-1]

    canvas = Canvas(image_size, len(class_list), len(class_list))
    success_matrix = np.zeros([len(class_list), len(class_list)]).astype(np.bool_)

    # Generate adversarial examples for each class
    for i, image in enumerate(images):

        # normalize
        x_val = np.reshape(image[:-1] * (1. / 255), [1, image_size, image_size, 1])
        class_id = image[-1]

        # Make example for each target class
        for j, target in enumerate(class_list):
            # Skip same class
            if i == j:
                canvas.paste_np_float(x_val, j, i, border=1)        # red border
                continue

            one_hot_target = np.zeros([1, num_classes], dtype=np.float32)
            one_hot_target[0, target] = 1.
            attack_params['y_target'] = one_hot_target
            adv_x_val = attack.generate_np(x_val, **attack_params)

            # Evaluate
            predict = cnn.sess.run(tf.argmax(probs, axis=1), feed_dict={x:adv_x_val})

            if predict[0] == target:
                border = 2
                success_matrix[i, j] = True
            elif predict[0] != class_id:
                border = 3
            else:
                border = None

            canvas.paste_np_float(adv_x_val, j, i, border=border)
        
    canvas.save(output_file)

    return success_matrix
