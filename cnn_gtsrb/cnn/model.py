import os
import logging
import tensorflow as tf
from cleverhans.model import Model
from cnn_gtsrb.dataset.base import DatasetProvider


logger = logging.getLogger('cnn')


class CNNModel(Model):

    def __init__(self, image_size, classes, model_name, model_dir,
                 kernel_size=[5, 5],
                 conv_layers=[32, 64],
                 fc_layer=1024,
                 ):
        """
        Make a CNN model.
        :param image_size: Image size of input.
        :param classes: Number of classes.
        :param model_name: model name (used for model saving).
        :param model_dir: Model save dir.
        :param kernel_size: Convolutional kernel size
        :param conv_layers: An array of the number of features of each layer.
        :param fc_layer: Number of features in the full connected layer.
        """

        self.image_size = image_size
        self.classes = classes
        self.conv_layers = conv_layers
        self.fc_layer = fc_layer
        self.kernel_size = kernel_size
        self.model_name = model_name
        self.model_dir = model_dir

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        self.params = {}

    def make_inputs(self):
        x = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 1], name='x')
        y = tf.placeholder(tf.float32, shape=[None, self.classes], name="y")

        return x, y

    def make_model(self, x):

        with tf.name_scope('cnn'):
            h_pool = x
            prev_layer_features = 1
            layer_size = self.image_size * self.image_size        # size of current layer

            for i, feature_count in enumerate(self.conv_layers):
                # Convolutional Layer
                with tf.name_scope('conv_{}'.format(i+1)):
                    W_conv = self.weight_variable(
                        [self.kernel_size[0], self.kernel_size[1], prev_layer_features, feature_count],
                        name='weight_{}'.format(i+1)
                    )

                    b_conv = self.bias_variable([feature_count], name='bias_{}'.format(i+1))
                    h_conv = tf.nn.relu(self.conv2d(h_pool, W_conv) + b_conv)

                with tf.name_scope('pool_{}'.format(i+1)):
                    # Pooling Layer #1 => 32 maps, 16x16
                    h_pool = self.max_pool_2x2(h_conv)

                prev_layer_features = feature_count
                layer_size //= 4


            # Full-connected Layer
            with tf.name_scope('fc1'):
                fc_size = layer_size * prev_layer_features
                W_fc1 = self.weight_variable([fc_size, self.fc_layer], name='fc_weight')
                b_fc1 = self.bias_variable([self.fc_layer], name='fc_bias')

                h_pool_flat = tf.reshape(h_pool, [-1, fc_size])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

            with tf.name_scope('dropout'):
                h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=0.5)

            with tf.name_scope('fc2'):
                # Readout layer
                W_fc2 = self.weight_variable([self.fc_layer, self.classes], name='fc_readout')
                b_fc2 = self.bias_variable([self.classes], name='bias_readout')

                probs = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

        self.create_global_step()

        return probs

    def fprop(self, x):
        probs = self.make_model(x)
        return {
            'probs': probs,
            'logits': probs,
        }
    
    def start_session(self):
        """Start a session that will be used for training."""
        self.sess = tf.Session()

        # Summary writers
        train_summary_dir = os.path.join(self.model_dir, 'training')
        test_summary_dir = os.path.join(self.model_dir, 'test')
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(test_summary_dir, self.sess.graph)

        return self.sess

    def end_session(self):
        self.sess.close()

    def train(self, probs, x, y, epoch, data_provider):
        """Train the model with given data.
        data_provider must be a subclass of DatasetProvider which provides a `next_batch` function
        that will return a tuple of (data, label).
        """
        if not issubclass(data_provider.__class__, DatasetProvider):
            raise TypeError('data_provider must be a subclass of DatasetProvider')

        with self.sess.as_default():

            global_step = self.create_global_step()

            # Loss function
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=probs)
            )

            # Train step
            train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)

            # Accuracy
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(probs, 1), tf.argmax(y, 1)), tf.float32)
            )

            summary = tf.summary.merge([
                tf.summary.scalar('loss', loss),
                tf.summary.scalar('accuracy', accuracy),
                tf.summary.image('input', x),
            ])

            self.sess.run(tf.global_variables_initializer())

            # Restore the model
            self.restore_model()

            for i in range(epoch):
                batch = data_provider.next_batch()

                train_step.run(
                    feed_dict={x: batch[0], y: batch[1]},
                    session=self.sess
                )

                if i % 100 == 0:
                    self.calculate_train_accuracy(accuracy, x, y, batch, summary=summary)

                if i % 1000 == 0:
                    self.calculate_test_accuracy(accuracy, x, y, data_provider.test_data(), summary=summary)

            # Save the model
            self.save_model()


    def calculate_train_accuracy(self, accuracy, x, y, batch, summary):
        accuracy_value, summary_value, step = self.sess.run(
            [accuracy, summary, self.create_global_step()],
            {x: batch[0], y: batch[1]}
        )
        print('Step {}, train accuracy = {}'.format(step, accuracy_value))
        self.train_summary_writer.add_summary(summary_value, step)

    def calculate_test_accuracy(self, accuracy, x, y, batch, summary):
        accuracy_value, summary_value, step = self.sess.run(
            [accuracy, summary, self.create_global_step()],
            {x: batch[0], y: batch[1]}
        )
        print('Step {}, test accuracy = {}'.format(step, accuracy_value))
        self.test_summary_writer.add_summary(summary_value, step)


    def adv_test(self, probs, x, y, adv_x, batch):
        with self.sess.as_default():

            # Accuracy
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(probs, 1), tf.argmax(y, 1)), tf.float32)
            )

            perturbation = tf.cast(((adv_x - x) + 1.) * (255 / 2), tf.uint8)

            summary = tf.summary.merge([
                tf.summary.scalar('adv_accuracy', accuracy),
                tf.summary.image('adv_input', adv_x),
                tf.summary.image('adv_perturbation', perturbation),
            ])

            self.sess.run(tf.global_variables_initializer())
            self.restore_model()

            self.calculate_test_accuracy(accuracy, x, y, batch, summary=summary)

    def init_session_and_restore(self):
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.restore_model()


    def save_model(self):
        saver = tf.train.Saver(self.params)
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        ckpt_file = '{}.ckpt'.format(self.model_name)
        ckpt_path = os.path.join(self.model_dir, ckpt_file)

        global_step = tf.train.get_global_step()

        save_path = saver.save(self.sess, ckpt_path, global_step=global_step)
        print('Model saved as {}.'.format(save_path))

    def restore_model(self):
        saver = tf.train.Saver(self.params)

        save_path = tf.train.latest_checkpoint(self.model_dir)
        if save_path:
            saver.restore(self.sess, save_path)

            global_step = tf.train.get_global_step()

            print('Model restored from {}, global_step={}'.format(save_path, global_step.eval(self.sess)))


    def weight_variable(self, shape, name):
        if name not in self.params:
            initial = tf.truncated_normal(shape, stddev=0.1)
            self.params[name] = tf.Variable(initial, name=name)

        return self.params[name]

    def bias_variable(self, shape, name):
        if name not in self.params:
            initial = tf.constant(0.1, shape=shape)
            self.params[name] = tf.Variable(initial, name=name)

        return self.params[name]

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def create_global_step(self):
        if 'global_step' not in self.params:
            self.params['global_step'] = tf.train.create_global_step()
        return self.params['global_step']