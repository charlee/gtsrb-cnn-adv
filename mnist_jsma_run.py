import logging
import tensorflow as tf
from cleverhans.attacks import SaliencyMapMethod
from cnn_gtsrb.dataset.mnist import MnistProvider
from cnn_gtsrb.cnn.model import CNNModel
#logging.basicConfig(level=logging.INFO)

tf.logging.set_verbosity(tf.logging.INFO)

jsma_params = {'theta': 1., 'gamma': 0.1, 'clip_min': 0., 'clip_max': 1., 'y_target': None}

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

jsma = SaliencyMapMethod(cnn, sess=cnn.sess)
adv_x = jsma.generate(x, **jsma_params)
probs = cnn.make_model(adv_x)

cnn.adv_test(probs, x, y, adv_x, mnist.test_data(size=100))
# cnn.test(mnist)
cnn.end_session()

#cnn.test(2000, mnist)

# for i in range(100):
#     data, label = gtsrb.next_batch('test')
#     print(data, label)


