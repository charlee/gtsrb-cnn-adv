import logging
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cnn_gtsrb.dataset.mnist import MnistProvider
from cnn_gtsrb.cnn.model import CNNModel
#logging.basicConfig(level=logging.INFO)

tf.logging.set_verbosity(tf.logging.INFO)

fgsm_params = {'eps': 0.5, 'clip_min': 0., 'clip_max': 1.}

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

x, y = cnn.make_inputs()

probs = cnn.make_model(x)
cnn.start_session()

fgsm = FastGradientMethod(cnn, sess=cnn.sess)
adv_x = fgsm.generate(x, **fgsm_params)
probs = cnn.make_model(adv_x)

cnn.adv_test(probs, x, y, adv_x, mnist.test_data(size=5000))
# cnn.test(mnist)
cnn.end_session()

#cnn.test(2000, mnist)

# for i in range(100):
#     data, label = gtsrb.next_batch('test')
#     print(data, label)


