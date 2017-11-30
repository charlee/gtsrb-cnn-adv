import glob
from PIL import Image
import numpy as np
import tensorflow as tf
from cnn_gtsrb.cnn.model import CNNModel
tf.logging.set_verbosity(tf.logging.INFO)


results = []
for f in glob.glob('tmp/adv_cgtsrb10-32x32/fast-jsma_*.npy'):
    results.append(np.load(f))

data = np.concatenate(results, axis=0)

cnn = CNNModel(
    image_size=32,
    classes=10,
    channels=3,
    model_name='cgtsrb10-32x32',
    model_dir='tmp/model-cgtsrb10-32x32',
    conv_layers=[32, 64, 128],
    fc_layers=[512],
)
# cnn = CNNModel(
#     image_size=32,
#     classes=10,
#     channels=3,
#     model_name='cifar10-32x32',
#     model_dir='tmp/model-cifar10-32x32',
#     kernel_size=[3, 3],
#     conv_layers=[48, 96, 192],
#     fc_layers=[512, 256],
# )

x, _ = cnn.make_inputs()
probs = cnn.make_model(x)

cnn.start_session()
cnn.init_session_and_restore()

orig_correct_count = 0
success_count = 0
actual_success_count = 0
actual_label_count = 0

for i, sample in enumerate(data):
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

    actual_orig_predict = cnn.sess.run(tf.argmax(probs, axis=1), feed_dict={x: image})
    if orig_predict == actual_orig_predict:
        orig_correct_count += 1

    if label == actual_orig_predict:
        actual_label_count += 1


    adv_x = np.reshape(adv_x, [1, image_size, image_size, channels])
    adv_x = adv_x * (1./255)

    actual_predict = cnn.sess.run(tf.argmax(probs, axis=1), feed_dict={x: adv_x})
    if actual_predict == target:
        actual_success_count += 1

    if adv_predict == target:
        success_count += 1

    # print('data ={}, actual={}'.format(sample[6:10], [label, actual_orig_predict[0], target, actual_predict[0]]))

    if i % 100 == 0:
        print('validating {}, lable = {}, orig = {}, success = {}, actual success = {}'.format(i, actual_label_count, orig_correct_count, success_count, actual_success_count))

    # if i > 10:
    #     break

print('success rate = {}, actual success rate = {}'.format(success_count / data.shape[0], actual_success_count / data.shape[0]))