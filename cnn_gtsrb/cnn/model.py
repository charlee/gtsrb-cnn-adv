import os
import logging
import tensorflow as tf
from cnn_gtsrb.dataset.base import DatasetProvider


logger = logging.getLogger('cnn')


class CNNModel():

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

    def make_model(self, x, y):

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
                # Dropout layer => [batch_size x 1024]
                keep_prob = tf.placeholder(tf.float32)
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


            with tf.name_scope('fc2'):
                # Readout layer
                W_fc2 = self.weight_variable([self.fc_layer, self.classes], name='fc_readout')
                b_fc2 = self.bias_variable([self.classes], name='bias_readout')

                probs = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Label
        with tf.name_scope('loss'):
            # Loss function
            cross_entrophy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=probs)
            )
            tf.summary.scalar('cross_entrophy_mean', cross_entrophy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(probs, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        self.x = x
        self.y = y
        self.keep_prob = keep_prob
        self.probs = probs
        self.cross_entrophy = cross_entrophy
        self.accuracy = accuracy

        self.merged = tf.summary.merge_all()

        tf.train.create_global_step()

    def start_session(self):
        """Start a session that will be used for training."""
        self.sess = tf.Session()
        return self.sess

    def end_session(self):
        self.sess.close()

    def train(self, epoch, data_provider):
        """Train the model with given data.
        data_provider must be a subclass of DatasetProvider which provides a `next_batch` function
        that will return a tuple of (data, label).
        """
        if not issubclass(data_provider.__class__, DatasetProvider):
            raise TypeError('data_provider must be a subclass of DatasetProvider')

        global_step = tf.train.get_global_step()

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entrophy, global_step=global_step)

        if not self.sess:
            self.start_session()

        # Summary writers
        train_summary_dir = os.path.join(self.model_dir, 'training')
        test_summary_dir = os.path.join(self.model_dir, 'test')
        if not os.path.isdir(train_summary_dir):
            os.makedirs(train_summary_dir)
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(test_summary_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # Restore the model
        self.restore_model()

        for i in range(epoch):
            batch = data_provider.next_batch()

            train_step.run(
                feed_dict={self.x: batch[0], self.y: batch[1], self.keep_prob: 0.5},
                session=self.sess
            )

            if i % 100 == 0:
                self.save_train_summary(batch)

            if i % 1000 == 0:
                test_batch = data_provider.test_data()
                self.save_test_summary(test_batch)

        # Save the model
        self.save_model()


    def test(self, data_provider):
        if not issubclass(data_provider.__class__, DatasetProvider):
            raise TypeError('data_provider must be a subclass of DatasetProvider')

        logger.info('Testing model ...')

        if not self.sess:
            self.start_session()

        self.sess.run(tf.global_variables_initializer())

        # Restore the model
        self.restore_model()

        # Evaluate the trainned model
        test_batch = data_provider.test_data()
        test_accuracy = self.accuracy.eval(
            feed_dict={self.x: test_batch[0], self.y: test_batch[1], self.keep_prob: 1.0},
            session=self.sess)
        print('test accuracy: {}'.format(test_accuracy))


    def save_train_summary(self, batch):
        """Save traning summary data (accuracy, cross_entrophy) to model dir."""
        global_step = tf.train.get_global_step()
        step = global_step.eval(self.sess)

        summary, train_accuracy = self.sess.run([self.merged, self.accuracy],
            feed_dict={self.x: batch[0], self.y: batch[1], self.keep_prob: 1.0})
        self.train_summary_writer.add_summary(summary, step)
        print('Step {}, training accuracy={}'.format(step, train_accuracy))

    def save_test_summary(self, batch):
        """Save test summary data (accuracy, cross_entrophy) to model dir."""
        global_step = tf.train.get_global_step()
        step = global_step.eval(self.sess)

        summary, test_accuracy = self.sess.run([self.merged, self.accuracy],
                                           feed_dict={self.x: batch[0], self.y: batch[1], self.keep_prob: 1.0})
        self.test_summary_writer.add_summary(summary, step)
        print('Step {}, test accuracy={}'.format(step, test_accuracy))



    def save_model(self):
        saver = tf.train.Saver()
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        ckpt_file = '{}.ckpt'.format(self.model_name)
        ckpt_path = os.path.join(self.model_dir, ckpt_file)

        global_step = tf.train.get_global_step()

        save_path = saver.save(self.sess, ckpt_path, global_step=global_step)
        logger.info('Model saved as {}.'.format(save_path))

    def restore_model(self):
        saver = tf.train.Saver()

        save_path = tf.train.latest_checkpoint(self.model_dir)
        if save_path:
            saver.restore(self.sess, save_path)

            global_step = tf.train.get_global_step()

            logger.info('Model restored from {}, global_step={}'.format(save_path, global_step.eval(self.sess)))


    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


