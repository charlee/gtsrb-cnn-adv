import logging
import random
import copy
import numpy as np
import math
import tensorflow as tf
from cnn_gtsrb.dataset.canvas import Canvas

from cleverhans.attacks_tf import jacobian_graph, apply_perturbations, saliency_map, jacobian
from cleverhans import utils_tf

from cleverhans.attacks import SaliencyMapMethod, FastGradientMethod

logger = logging.getLogger('craft.py')
if len(logger.handlers) == 0:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s %(name)s] ' +
                                    '%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

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

    def __init__(self, cnn_model, attack_params, image_size, num_classes, channels):
        """
        :param x: original input tensor.
        :param cnn_model: cnn model class.
        """
        self.cnn_model = cnn_model
        self.attack_params = attack_params
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels

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

        data = np.reshape(batch[:,:-1], [-1, self.image_size, self.image_size, self.channels])
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
            np.reshape(adv_examples * 255, [-1, self.image_size * self.image_size * self.channels]),
            y,
            adv_predicts
        )

    def summarize(self, batch, legit_predicts, adv_examples, targets, adv_predicts):
        raise NotImplemented()


class BatchFGSMCrafting(BatchCrafting):
    name = 'fgsm'
    attack_class = FastGradientMethod

    def __init__(self, cnn_model, attack_params, *args, **kwargs):
        assert 'eps' in attack_params, 'attack_params MUST contain `eps`'
        super().__init__(cnn_model, attack_params, *args, **kwargs)

    def summarize(self, batch, legit_predicts, adv_examples, targets, adv_predicts):
        result = []

        batch_size = batch.shape[0]

        # #0 column: image size    
        result.append(np.full([batch_size, 1], self.image_size))
        # #1 column: num of classes
        result.append(np.full([batch_size, 1], self.num_classes))
        # #2: channels
        result.append(np.full([batch_size, 1], self.channels))

        # 3rd column: epsillon
        result.append(np.full([batch_size, 1], self.attack_params['eps']))
        # 4-5th column: reserved
        result.append(np.zeros([batch_size, 2]))
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
    name = 'jsma'
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
        # #2: channels
        result.append(np.full([batch_size, 1], self.channels))
        # #2: gamma
        result.append(np.full([batch_size, 1], self.attack_params['gamma']))
        # #3: gamma
        result.append(np.full([batch_size, 1], self.attack_params['theta']))
        # #4-#5: reserved
        result.append(np.zeros([batch_size, 1]))
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


class FastBatchJSMACrafting(BatchCrafting):
    name = 'fast-jsma'
    attack_class = SaliencyMapMethod

    def __init__(self, cnn_model, attack_params, *args, **kwargs):
        super().__init__(cnn_model, attack_params, *args, **kwargs)

    def jsma(self, sess, x, predictions, grads, sample, target, theta, gamma, clip_min,
            clip_max, feed=None):
        """
        TensorFlow implementation of the JSMA (see https://arxiv.org/abs/1511.07528
        for details about the algorithm design choices).
        :param sess: TF session
        :param x: the input placeholder
        :param predictions: the model's symbolic output (the attack expects the
                    probabilities, i.e., the output of the softmax, but will
                    also work with logits typically)
        :param grads: symbolic gradients
        :param sample: numpy array with sample input
        :param target: target class for sample input
        :param theta: delta for each feature adjustment
        :param gamma: a float between 0 - 1 indicating the maximum distortion
            percentage
        :param clip_min: minimum value for components of the example returned
        :param clip_max: maximum value for components of the example returned
        :return: an adversarial sample
        """

        # Copy the source sample and define the maximum number of features
        # (i.e. the maximum number of iterations) that we may perturb
        adv_x = copy.copy(sample)
        # count the number of features. For MNIST, 1x28x28 = 784; for
        # CIFAR, 3x32x32 = 3072; etc.
        nb_features = np.product(adv_x.shape[1:])
        # reshape sample for sake of standardization
        original_shape = adv_x.shape
        adv_x = np.reshape(adv_x, (1, nb_features))
        # compute maximum number of iterations
        max_iters = np.floor(nb_features * gamma / 2)

        # Find number of classes based on grads
        nb_classes = len(grads)

        increase = bool(theta > 0)

        # Compute our initial search domain. We optimize the initial search domain
        # by removing all features that are already at their maximum values (if
        # increasing input features---otherwise, at their minimum value).
        if increase:
            search_domain = set([i for i in range(nb_features)
                                if adv_x[0, i] < clip_max])
        else:
            search_domain = set([i for i in range(nb_features)
                                if adv_x[0, i] > clip_min])

        # Initialize the loop variables
        iteration = 0
        adv_x_original_shape = np.reshape(adv_x, original_shape)
        current = utils_tf.model_argmax(sess, x, predictions, adv_x_original_shape,
                                        feed=feed)

        # charlee: Used to log when the model gets confused
        orig_label = current
        confused_at = 0
        success_at = 0

        logger.debug("Starting JSMA attack up to {} iterations".format(max_iters))
        # Repeat this main loop until we have achieved misclassification
        while (current != target and iteration < max_iters and
            len(search_domain) > 1):
            # Reshape the adversarial example
            adv_x_original_shape = np.reshape(adv_x, original_shape)

            # Compute the Jacobian components
            grads_target, grads_others = jacobian(sess, x, grads, target,
                                                adv_x_original_shape,
                                                nb_features, nb_classes,
                                                feed=feed)

            if iteration % ((max_iters + 1) // 5) == 0 and iteration > 0:
                logger.debug("Iteration {} of {}".format(iteration,
                                                        int(max_iters)))
            # Compute the saliency map for each of our target classes
            # and return the two best candidate features for perturbation
            i, j, search_domain = saliency_map(
                grads_target, grads_others, search_domain, increase)

            # Apply the perturbation to the two input features selected previously
            adv_x = apply_perturbations(
                i, j, adv_x, increase, theta, clip_min, clip_max)

            # Update our current prediction by querying the model
            current = utils_tf.model_argmax(sess, x, predictions,
                                            adv_x_original_shape, feed=feed)

            # Update loop variables
            iteration = iteration + 1

            # charlee: Record the iternation when model gets confused
            if current != orig_label and confused_at == 0:
                confused_at = iteration

        if current == target:
            logger.info("Attack succeeded using {} iterations".format(iteration))
            success_at = iteration
        else:
            logger.info(("Failed to find adversarial example " +
                        "after {} iterations").format(iteration))

        # Compute the ratio of pixels perturbed by the algorithm
        percent_perturbed = float(iteration * 2) / nb_features
        confused_at = float(confused_at * 2) / nb_features
        success_at = float(success_at * 2) / nb_features

        # Report success when the adversarial example is misclassified in the
        # target class
        return np.reshape(adv_x, original_shape), percent_perturbed, confused_at, success_at, orig_label, current


    def batch_jsma_with_perturbation_rate(self, batch, max_gamma, theta):

        # Make random targets

        x, _ = self.cnn_model.make_inputs()
        preds = self.cnn_model.get_probs(x)
        grads = jacobian_graph(preds, x, self.num_classes)

        results = []

        for i, data in enumerate(batch):

            sample = np.reshape(data[:-1], [self.image_size, self.image_size, self.channels]) * (1. / 255)
            sample = np.expand_dims(sample, axis=0)
            target = data[-1]
            while target == data[-1]:
                target = random.randint(0, self.num_classes - 1)

            # Use JSMA and try to make one single adv example for given sample
            adv_x, perturbation_rate, confused_at, success_at, orig_predict, adv_predict = self.jsma(
                sess=self.cnn_model.sess,
                x=x,
                predictions=preds,
                grads=grads,
                sample=sample,
                target=target,
                theta=theta,
                gamma=max_gamma,
                clip_min=0.,
                clip_max=1.,
            )

            print('perturbation_rate={}, confused_at={}, success_at={}'.format(perturbation_rate, confused_at, success_at))
        
            results.append({
                'adv_x': adv_x,
                'perturbation_rate': perturbation_rate,
                'target': target,
                'confused_at': confused_at,
                'success_at': success_at,
                'orig_predict': orig_predict,
                'adv_predict': adv_predict,
            })

        return results

    def summarize(self, batch, adv_data):
        """
        adv_data is the return value from batch_jsma_with_perturbation_rate.
        """
        results = []

        for i, input_data in enumerate(batch):
            result = []
            result.append(self.image_size)          #0
            result.append(self.num_classes)         #1
            result.append(self.channels)            #2
            result.append(1.0)                      #3 theta
            result.append(adv_data[i]['confused_at'])   #4
            result.append(adv_data[i]['success_at'])    #5
            result.append(input_data[-1])               #6: label
            result.append(adv_data[i]['orig_predict'])  #7: orig predict
            result.append(adv_data[i]['target'])        #8: target
            result.append(adv_data[i]['adv_predict'])   #9: adv_predict
            result += input_data[:-1].tolist()      #10.. original image
            result += (adv_data[i]['adv_x'].flatten() * 255).tolist()

            results.append(result)

        return np.array(results, dtype=np.float32)

