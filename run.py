import os
import sys
import logging
# import tensorflow as tf
from cnn_gtsrb.dataset.color_gtsrb_10 import ColorGtsrb10Provider
from cnn_gtsrb.dataset.fashion_mnist import FashionMnistProvider
from cnn_gtsrb.dataset.mnist_bg import MnistBgProvider
from cnn_gtsrb.dataset.cifar10 import Cifar10Provider
from cnn_gtsrb.cnn.model import CNNModel
#logging.basicConfig(level=logging.INFO)

from cnn_gtsrb.attacks.crafting import BatchFGSMCrafting, BatchJSMACrafting

class ExperimentBase():

    MODEL_NAME = ''

    def __init__(self):
        self.adv_dir = os.path.join('tmp', 'adv_{}'.format(self.MODEL_NAME))

    def train(self, epoch=20000):
        print('training {} for epoch={}'.format(self.MODEL_NAME, epoch))
        x, y = self.cnn.make_inputs()
        probs = self.cnn.make_model(x)

        self.cnn.start_session()
        self.cnn.train(probs, x, y, epoch, self.dataset)
        # cnn.test(gtsrb)
        self.cnn.end_session()

    def craft_adv(self, num_data, attack, var_name, var_values):
        """Crafting adversarial examples.
        attack: Attack algorithm.
        params_list: a list of adversarial parameters.
        """
        test_data = self.dataset.raw_test_data()[:num_data]
        batch_size = 100

        if not os.path.isdir(self.adv_dir):
            os.makedirs(self.adv_dir)

        self.cnn.start_session()
        self.cnn.init_session_and_restore()

        for var_value in var_values:
            attack.update_params({var_name: var_value})
            for batch_pos in range(0, test_data.shape[0], batch_size):

                filepath = os.path.join(self.adv_dir, '{}_{}-{:0.2f}-{}.npy'.format(
                    attack.name, self.dataset.name, var_value, batch_pos))

                if os.path.isfile(filepath):
                    print('{} exists, skip this batch'.format(filepath))

                else:
                    print("======= batch {}, {} = {}".format(batch_pos, var_name, var_value))
                    batch = test_data[batch_pos:batch_pos+batch_size]
                    result = attack.craft_examples(batch)
                    result = attack.summarize(batch, *result)

                    result.dump(filepath)

        self.cnn.end_session()

    def craft_fgsm(self, num_data):
        params = {'eps': 0.2, 'clip_min': 0., 'clip_max': 1.}
        attack = BatchFGSMCrafting(self.cnn, params, self.dataset.IMAGE_SIZE, self.dataset.CLASSES, self.dataset.CHANNELS)
        var_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        self.craft_adv(num_data, attack, 'eps', var_values)

    def craft_jsma(self, num_data):
        params = {'theta': 1., 'gamma': 0.1, 'clip_min': 0., 'clip_max': 1., 'y_target': None}
        attack = BatchJSMACrafting(self.cnn, params, self.dataset.IMAGE_SIZE, self.dataset.CLASSES, self.dataset.CHANNELS)
        var_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12]

        self.craft_adv(num_data, attack, 'gamma', var_values)



# tf.logging.set_verbosity(tf.logging.INFO)


class CGTSRB10(ExperimentBase):

    MODEL_NAME = 'cgtsrb10-32x32'

    def __init__(self):
        super().__init__()

        self.dataset = ColorGtsrb10Provider()
        self.cnn = CNNModel(
            image_size=self.dataset.IMAGE_SIZE,
            classes=self.dataset.CLASSES,
            channels=self.dataset.CHANNELS,
            model_name=self.MODEL_NAME,
            model_dir='tmp/model-{}'.format(self.MODEL_NAME),
            conv_layers=[32, 64, 128],
            fc_layers=[512],
        )


class CIFAR10(ExperimentBase):
    MODEL_NAME = 'cifar10-32x32'

    def __init__(self):
        super().__init__()

        self.dataset = Cifar10Provider()
        self.cnn = CNNModel(
            image_size=self.dataset.IMAGE_SIZE,
            classes=self.dataset.CLASSES,
            channels=self.dataset.CHANNELS,
            kernel_size=[3, 3],
            model_name=self.MODEL_NAME,
            model_dir='tmp/model-{}'.format(self.MODEL_NAME),
            conv_layers=[48, 96, 192],
            fc_layers=[512, 256],
        )


class FashionMNIST(ExperimentBase):
    MODEL_NAME = 'fmnist-28x28'

    def __init__(self):
        super().__init__()
        self.dataset = FashionMnistProvider()
        self.cnn = CNNModel(
            image_size=self.dataset.IMAGE_SIZE,
            classes=self.dataset.CLASSES,
            channels=self.dataset.CHANNELS,
            model_name=self.MODEL_NAME,
            model_dir='tmp/model-{}'.format(self.MODEL_NAME),
            conv_layers=[32, 64],
            fc_layers=[1024],
        )


class MNISTBG(ExperimentBase):
    MODEL_NAME = 'mnist_bg-28x28'

    def __init__(self):
        super().__init__()
        self.dataset = MnistBgProvider()

        self.cnn = CNNModel(
            image_size=self.dataset.IMAGE_SIZE,
            classes=self.dataset.CLASSES,
            channels=self.dataset.CHANNELS,
            model_name=self.MODEL_NAME,
            model_dir='tmp/model-{}'.format(self.MODEL_NAME),
            conv_layers=[32, 64],
            fc_layers=[1024],
        )


if __name__ == '__main__':
    cmd = sys.argv[1]
    model_name = sys.argv[2]

    if model_name == 'cgtsrb10':
        model = CGTSRB10()
    elif model_name == 'fmnist':
        model = FashionMNIST()
    elif model_name == 'mnistbg':
        model = MNISTBG()
    elif model_name == 'cifar10':
        model = CIFAR10()

    if cmd == 'train':
        model.train(20000)
    elif cmd == 'adv_fgsm':
        model.craft_fgsm(5000)
    elif cmd == 'adv_jsma':
        model.craft_jsma(2500)