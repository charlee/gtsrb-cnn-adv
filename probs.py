import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn_gtsrb.cnn.model import CNNModel


data = np.load('tmp/adv_mnist-28x28/fast-jsma_mnist-0.npy')

cnn = CNNModel(
    image_size=28,
    classes=10,
    channels=1,
    model_name='mnist-28x28',
    model_dir='tmp/model-mnist-28x28',
    conv_layers=[32, 64],
    fc_layers=[1024],
)

x, _ = cnn.make_inputs()
probs = cnn.make_model(x)

cnn.start_session()
cnn.init_session_and_restore()

orig_correct_count = 0
success_count = 0
actual_success_count = 0
actual_label_count = 0

sample = data[0]

legit_preds = []
adv_preds = []

iteration = 1000
for i in range(iteration):
    # sample = data[0]
    image_size = int(sample[0])
    num_classes = int(sample[1])
    channels = int(sample[2])

    label = sample[6]
    orig_predict = sample[7]
    target = sample[8]
    adv_predict = sample[9]

    size = int(image_size*image_size*channels)
    image = sample[10:10+size]
    adv_x = sample[10+size:10+size+size]

    # im = Image.fromarray(np.reshape(adv_x, [image_size, image_size]))
    # im.show()

    image = np.reshape(image, [1, image_size, image_size, channels])
    image = image * (1./255)

    legit_preds.append(cnn.sess.run(probs, feed_dict={x: image}))

    adv_x = np.reshape(adv_x, [1, image_size, image_size, channels])
    adv_x = adv_x * (1./255)

    adv_preds.append(cnn.sess.run(probs, feed_dict={x: adv_x}))


legit = np.mean(legit_preds, axis=0)
adv = np.mean(adv_preds, axis=0)

plt.plot(range(num_classes), legit[0], label='legit:{}'.format(orig_predict))
plt.plot(range(num_classes), adv[0], label='adv:{}'.format(adv_predict))
plt.title('Softmax probs on MNIST ({} average)'.format(iteration))


plt.legend()
plt.show()
