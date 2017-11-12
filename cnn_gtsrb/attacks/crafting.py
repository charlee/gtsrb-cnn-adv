import logging
import copy
import numpy as np
import math
import tensorflow as tf
from cnn_gtsrb.dataset.canvas import Canvas

from cleverhans.attacks import SaliencyMapMethod, FastGradientMethod

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


class BatchCrafting():

    attack_class = SaliencyMapMethod

    def __init__(self, cnn_model, attack_params, image_size, num_classes):
        """
        :param x: original input tensor.
        :param cnn_model: cnn model class.
        """
        self.cnn_model = cnn_model
        self.attack_params = attack_params
        self.num_classes = num_classes
        self.image_size = image_size

        # Input and Labels
        # x = [batch, size, size, 1], y = [batch, classes]
        self.x, self.y = self.cnn_model.make_inputs()

        # Predict output (one-hot)
        # probs = [batch, classes]
        self.probs = self.cnn_model.make_model(self.x)

        # Attach algorithm
        self.attack = self.attack_class(self.cnn_model)

    def update_params(self, params):
        """Update part of the attack params. (i.e. the variable parts)"""
        self.attack_params.update(params)

    def craft_examples(self, batch):
        """Create adversarial examples for the batch.
        Batch should be [None, size*size+1], in which [:,0..-1] are the images
        and [:-1] are the labels.

        Return: a tuple of (legit_predicts ,adv_examples, targeted_classes, adv_predicts)
        """
        sess = self.cnn_model.sess
        self.attack.sess = sess

        # Make random targets
        y = np.random.randint(self.num_classes, size=batch.shape[0])
        y_one_hot = np.zeros([batch.shape[0], self.num_classes])
        for idx, label in enumerate(y):
            y_one_hot[idx, label] = 1

        # Adversarial input
        # adv_x = [batch, size, size, 1]
        params = copy.copy(self.attack_params)
        params.update({'y_target': y_one_hot})
        # adv_x = self.attack.generate(self.x, **params)

        # Craft adversarial examples

        data = np.reshape(batch[:,:-1], [-1, self.image_size, self.image_size, 1])
        data = data * (1./255)

        legit_predicts = sess.run(
            tf.argmax(self.probs, axis=1), 
            feed_dict={self.x: data},
        )

        
        # adv_examples = sess.run(adv_x, feed_dict={self.x: data})
        adv_examples = self.attack.generate_np(data, **params)

        # Adversarial predict output    
        adv_predicts = sess.run(
            tf.argmax(self.probs, axis=1),
            feed_dict={self.x: adv_examples},
        )

        return (
            legit_predicts,
            np.reshape(adv_examples * 255, [-1, self.image_size * self.image_size]),
            y,
            adv_predicts
        )

    def summarize(self, batch, legit_predicts, adv_examples, targets, adv_predicts):
        raise NotImplemented()

class BatchFGSMCrafting(BatchCrafting):
    attack_class = FastGradientMethod

    def __init__(self, cnn_model, attack_params, *args, **kwargs):
        assert 'eps' in attack_params, 'attack_params MUST contain `eps`'
        super().__init__(cnn_model, attack_params, *args, **kwargs)

    def summarize(self, batch, legit_predicts, adv_examples, targets, adv_predicts):
        result = []

        batch_size = batch.shape[0]

        # 1st column: image size    
        result.append(np.full([batch_size, 1], self.image_size))
        # 2nd column: num of classes
        result.append(np.full([batch_size, 1], self.num_classes))
        # 3rd column: epsillon
        result.append(np.full([batch_size, 1], self.attack_params['eps']))
        # 4-6th column: reserved
        result.append(np.zeros([batch_size, 3]))
        # 7th column: labels (correct classes)
        result.append(batch[:,-1:])
        # 8th column: legit predicts
        result.append(np.expand_dims(legit_predicts, 1))
        # 9th column: adversarial targets
        result.append(np.expand_dims(targets, 1))
        # 10th column: adversarial predicts
        result.append(np.expand_dims(adv_predicts, 1))
        # original images
        result.append(batch[:,:-1])
        # adversarial examples
        result.append(adv_examples)

        return np.concatenate(result, axis=1)


class BatchJSMACrafting(BatchCrafting):
    attack_class = SaliencyMapMethod

    def __init__(self, cnn_model, attack_params, *args, **kwargs):
        assert 'gamma' in attack_params, 'attack_params MUST contain `gamma`'
        assert 'theta' in attack_params, 'attack_params MUST contain `theta`'
        super().__init__(cnn_model, attack_params, *args, **kwargs)

    def summarize(self, batch, legit_predicts, adv_examples, targets, adv_predicts):
        result = []

        batch_size = batch.shape[0]

        # #0: image size    
        result.append(np.full([batch_size, 1], self.image_size))
        # #1: num of classes
        result.append(np.full([batch_size, 1], self.num_classes))
        # #2: gamma
        result.append(np.full([batch_size, 1], self.attack_params['gamma']))
        # #3: gamma
        result.append(np.full([batch_size, 1], self.attack_params['theta']))
        # #4-#5: reserved
        result.append(np.zeros([batch_size, 2]))
        # #6: labels (correct classes)
        result.append(batch[:,-1:])
        # #7: legit predicts
        result.append(np.expand_dims(legit_predicts, 1))
        # #8: adversarial targets
        result.append(np.expand_dims(targets, 1))
        # #9: adversarial predicts
        result.append(np.expand_dims(adv_predicts, 1))
        # original images
        result.append(batch[:,:-1])
        # adversarial examples
        result.append(adv_examples)

        return np.concatenate(result, axis=1)