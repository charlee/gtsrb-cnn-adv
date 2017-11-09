import os
import logging
import tensorflow as tf
from cnn_gtsrb.dataset.base import DatasetProvider


logger = logging.getLogger('cnn')


class CNNModel():

    def __init__(self, image_size, classes, model_name, model_dir):

        self.image_size = image_size
        self.classes = classes
        self.model_name = model_name
        self.model_dir = model_dir

        # Input layer
        x = tf.placeholder(tf.int8, shape=[None, self.image_size * self.image_size], name='raw_input')

        # Convert to float32 matrix of [batch_size, self.image_size, self.image_size, color_depth]
        input = tf.cast(x, tf.float32) * (1. /255) - 0.5
        input = tf.reshape(input, shape=[-1, self.image_size, self.image_size, 1], name='reshaped_input')

        # Convolutional Layer #1 => 32 maps, 32x32
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(input, W_conv1) + b_conv1, name='conv1')

        # Pooling Layer #1 => 32 maps, 16x16
        h_pool1 = self.max_pool_2x2(h_conv1, name='pool1')

        # Convolutional Layer #2 => 64 maps, 16x16
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2, name='conv2')

        # Polling Layer #2 => 64 maps, 8x8
        h_pool2 = self.max_pool_2x2(h_conv2, name='pool2')

        # Full-connected Layer
        image_size2 = self.image_size // 4
        fc_size = image_size2 * image_size2 * 64
        W_fc1 = self.weight_variable([fc_size, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, fc_size])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='fc1')

        # Dropout layer => [batch_size x 1024]
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Readout layer
        W_fc2 = self.weight_variable([1024, self.classes])
        b_fc2 = self.bias_variable([self.classes])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Label
        y_ = tf.placeholder(tf.float32, shape=[None, self.classes], name="labels")

        with tf.name_scope('cross_entrophy'):
            # Loss function
            cross_entrophy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
            )
            tf.summary.scalar('cross_entrophy', cross_entrophy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        self.x = x
        self.y_ = y_
        self.keep_prob = keep_prob
        self.y_conv = y_conv
        self.cross_entrophy = cross_entrophy
        self.accuracy = accuracy

        self.merged = tf.summary.merge_all()

        tf.train.create_global_step()


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

        with tf.Session() as sess:
            # Summary writers
            train_summary_dir = os.path.join(self.model_dir, 'training')
            test_summary_dir = os.path.join(self.model_dir, 'test')
            if not os.path.isdir(train_summary_dir):
                os.makedirs(train_summary_dir)
            self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            self.test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            sess.run(tf.global_variables_initializer())

            # Restore the model
            self.restore_model(sess)

            for i in range(epoch):
                batch = data_provider.next_batch()

                train_step.run(
                    feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})

                if i % 100 == 0:
                    self.save_train_summary(sess, batch)

                if i % 1000 == 0:
                    test_batch = data_provider.next_batch('test', 1000)
                    self.save_test_summary(sess, test_batch)


            # Save the model
            self.save_model(sess)


    def test(self, batch_size, data_provider):
        if not issubclass(data_provider.__class__, DatasetProvider):
            raise TypeError('data_provider must be a subclass of DatasetProvider')

        logger.info('Testing model with {} data...'.format(batch_size))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Restore the model
            self.restore_model(sess)

            # Evaluate the trainned model
            test_batch = data_provider.next_batch(type='test', batch_size=batch_size)
            test_accuracy = self.accuracy.eval(
                feed_dict={self.x: test_batch[0], self.y_: test_batch[1], self.keep_prob: 1.0})
            print('test accuracy: {}'.format(test_accuracy))


    def save_train_summary(self, sess, batch):
        """Save traning summary data (accuracy, cross_entrophy) to model dir."""
        global_step = tf.train.get_global_step()
        step = global_step.eval(sess)

        summary, train_accuracy = sess.run([self.merged, self.accuracy],
            feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
        self.train_summary_writer.add_summary(summary, step)
        print('Step {}, training accuracy={}'.format(step, train_accuracy))

    def save_test_summary(self, sess, batch):
        """Save test summary data (accuracy, cross_entrophy) to model dir."""
        global_step = tf.train.get_global_step()
        step = global_step.eval(sess)

        summary, test_accuracy = sess.run([self.merged, self.accuracy],
                                           feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
        self.test_summary_writer.add_summary(summary, step)
        print('Step {}, test accuracy={}'.format(step, test_accuracy))



    def save_model(self, sess):
        saver = tf.train.Saver()
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        ckpt_file = '{}.ckpt'.format(self.model_name)
        ckpt_path = os.path.join(self.model_dir, ckpt_file)

        global_step = tf.train.get_global_step()

        save_path = saver.save(sess, ckpt_path, global_step=global_step)
        logger.info('Model saved as {}.'.format(save_path))

    def restore_model(self, sess):
        saver = tf.train.Saver()

        save_path = tf.train.latest_checkpoint(self.model_dir)
        if save_path:
            saver.restore(sess, save_path)

            global_step = tf.train.get_global_step()

            logger.info('Model restored from {}, global_step={}'.format(save_path, global_step.eval(sess)))


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x, name=None):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME', name=name)


