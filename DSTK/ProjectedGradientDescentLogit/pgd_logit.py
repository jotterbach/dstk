import tensorflow as tf
import numpy as np
import sys


class ProjectedGradientDescentLogit(object):
    '''The projected Gradient Descent optimizer does the trick!'''

    def __init__(self, n_input, l1=0.0, l2=0.0):

        self.l1 = l1
        self.l2 = l2

        self._recording = {'epoch': 0,
                           'batch_cost': []}

        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32, [None, 1])

        self.W = tf.Variable(tf.random_uniform([n_input, 1], minval=1e-2, maxval=.1))
        self.b = tf.Variable(tf.zeros([1]))

        self.cost = self._loss()
        self.optimizer = tf.train.FtrlOptimizer(learning_rate=self.learning_rate, l1_regularization_strength=self.l1, l2_regularization_strength=self.l2).minimize(self.cost)
        self.clipper = self._clipper()
        self.probas = self._predict_proba()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        num_cores = 6
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'CPU': num_cores},
                                                                inter_op_parallelism_threads=num_cores,
                                                                intra_op_parallelism_threads=num_cores))
        self.sess.run(init)

    def partial_fit(self, X, y, learning_rate=None):

        if not learning_rate:
            self.learning_rate = 0.001

        # update with a gradient step
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X,
                                             self.y: y,
                                             self.learning_rate: learning_rate})

        # project to feasible set
        self.sess.run(self.clipper)

        return cost

    def predict_proba(self, X):
        return self.sess.run(self.probas, feed_dict={self.x: X})

    def _predict_proba(self):
        return tf.nn.sigmoid(tf.matmul(self.x, self.W) + self.b, name="sigmoid")

    def _loss(self):
        return tf.nn.sigmoid_cross_entropy_with_logits(tf.matmul(self.x, self.W) + self.b, self.y, name="log_loss")

    def _clipper(self):
        '''
        projects the weights to the feasible set
        :return:
        '''
        return tf.assign(self.W, tf.clip_by_value(self.W, 0, np.PINF), name="projector")

    def train(self, X, y, n_epochs, learning_rate=0.0001, display_step=10, batch_size=None):

        data = DataSet(X, y, batch_size)

        for epoch in range(n_epochs):

            self._recording['epoch'] += 1

            data.reset_counter()
            costs = []

            while data.has_next():
                batch_xs, batch_ys = data.next_batch()
                costs.append(self.partial_fit(batch_xs, batch_ys, learning_rate))

            self._recording['batch_cost'].append(np.mean(costs))

            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                sys.stdout.write("\r>> Epoch: {0:04d} / {1:04d}, cost={2:.9f}".format(epoch + 1, n_epochs, self._recording['batch_cost'][-1]))
                sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()


class DataSet(object):

    def __init__(self, data, target, batch_size=None):
        self.data = data
        self.target = target
        self.n_samples = self.data.shape[0]
        self.n_dimensions = self.data.shape[1]
        if batch_size is None:
            self.batch_size = self.n_samples
        else:
            self.batch_size = batch_size
        self.total_batches = int(self.n_samples / self.batch_size)
        self.current_batch = 0

    def next_batch(self):
        batch_lower_idx = self.current_batch * self.batch_size
        batch_upper_idx = (self.current_batch + 1) * self.batch_size

        self.current_batch += 1

        return self.data[batch_lower_idx:batch_upper_idx, :], self.target[batch_lower_idx:batch_upper_idx].reshape(self.batch_size, 1)

    def has_next(self):
        return self.current_batch < self.total_batches

    def reset_counter(self):

        rand_idx = np.random.permutation(range(self.n_samples))
        self.data = self.data[rand_idx, :]
        self.target = self.target[rand_idx]
        self.current_batch = 0
